/**
 * Gear-Spline LIO Node
 *
 * Milestone 1: Basic IESKF validation with uniform B-spline (dt=0.05s, NORMAL mode)
 * Based on RESPLE, converted from ROS2 to ROS1
 */

#include <ros/ros.h>
#include <std_msgs/Int64.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Livox ROS driver support
#include <livox_ros_driver/CustomMsg.h>

#include <queue>
#include <thread>
#include <mutex>
#include <sstream>
#include <iomanip>

#include "gear_spline/Estimator.hpp"
#include "gear_spline/GearSystem.hpp"

// Global variables (following RESPLE pattern)
int NUM_OF_THREAD = 4;
int NUM_MATCH_POINTS = 5;
KD_TREE<pcl::PointXYZINormal> ikdtree;

class GearSplineLIO
{
public:
    GearSplineLIO(ros::NodeHandle& nh) : nh_(nh)
    {
        readParameters();

        // IMU subscriber
        if (!if_lidar_only_) {
            sub_imu_ = nh_.subscribe(topic_imu_, 2000, &GearSplineLIO::imuCallback, this);
        }

        // LiDAR subscriber based on type
        if (lidar_type_ == "Livox" || lidar_type_ == "Mid70Avia") {
            sub_livox_ = nh_.subscribe(topic_lidar_, 200, &GearSplineLIO::livoxLidarCallback, this);
            ROS_INFO("Using Livox LiDAR format");
        } else {
            sub_lidar_ = nh_.subscribe(topic_lidar_, 200, &GearSplineLIO::ousterLidarCallback, this);
            ROS_INFO("Using Ouster LiDAR format");
        }

        // Publishers
        pub_odom_ = nh_.advertise<nav_msgs::Odometry>("odometry", 50);
        pub_path_ = nh_.advertise<nav_msgs::Path>("path", 50);
        pub_cur_scan_ = nh_.advertise<sensor_msgs::PointCloud2>("current_scan", 2);
        pub_start_time_ = nh_.advertise<std_msgs::Int64>("start_time", 50);
        pub_gear_marker_ = nh_.advertise<visualization_msgs::Marker>("gear_mode", 1);

        pc_last_.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        pc_last_ds_.reset(new pcl::PointCloud<pcl::PointXYZINormal>());

        ROS_INFO("GearSplineLIO initialized with dt=%.3fs (%.1fHz knot frequency)",
                 double(dt_ns_) * 1e-9, 1e9 / double(dt_ns_));
    }

    void processData()
    {
        ros::Rate rate(20);
        int64_t max_spl_knots = 0;
        int64_t t_last_map_upd = 0;

        while (ros::ok()) {
            // Process LiDAR data from buffer
            while (!t_buff_.empty()) {
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_frame(new pcl::PointCloud<pcl::PointXYZINormal>());

                mtx_pc_.lock();
                pc_frame->points = pc_buff_.front();
                pc_buff_.pop_front();
                int64_t time_begin = t_buff_.front();
                t_buff_.pop_front();
                mtx_pc_.unlock();

                std::vector<int> indices;
                pcl::removeNaNFromPointCloud(*pc_frame, *pc_frame, indices);
                pc_last_ds_->clear();

                ds_filter_body_.setInputCloud(pc_frame);
                ds_filter_body_.filter(*pc_last_ds_);
                std::sort(pc_last_ds_->points.begin(), pc_last_ds_->points.end(), &CommonUtils::time_list);

                for (size_t i = 0; i < pc_last_ds_->points.size(); i++) {
                    PointData pt(pc_last_ds_->points[i], time_begin, q_bl_, t_bl_, w_pt_);
                    pt_buff_.push_back(pt);
                }
            }

            // Process IMU data
            if (!if_lidar_only_ && !imu_int_buff_.empty()) {
                m_buff_.lock();
                std::vector<sensor_msgs::Imu::ConstPtr> imu_buff_msg = imu_int_buff_;
                imu_int_buff_.clear();
                m_buff_.unlock();

                for (size_t i = 0; i < imu_buff_msg.size(); i++) {
                    const auto& imu_msg = imu_buff_msg[i];
                    int64_t t_ns = imu_msg->header.stamp.toNSec();
                    Eigen::Vector3d acc(imu_msg->linear_acceleration.x,
                                       imu_msg->linear_acceleration.y,
                                       imu_msg->linear_acceleration.z);
                    if (acc_ratio_) acc *= 9.81;
                    Eigen::Vector3d gyro(imu_msg->angular_velocity.x,
                                        imu_msg->angular_velocity.y,
                                        imu_msg->angular_velocity.z);
                    ImuData imu(t_ns, gyro, acc);
                    imu_buff_.push_back(imu);
                }
            }

            if (!initialization()) {
                rate.sleep();
                continue;
            }

            while (collectMeasurements()) {
                int64_t max_time_ns = pt_meas_.back().time_ns;

                if (if_lidar_only_) {
                    estimator_lo_.propRCP(max_time_ns);
                    estimator_lo_.updateIEKFLiDAR(pt_meas_, &ikdtree, param_.nn_thresh, param_.coeff_cov);
                } else {
                    if (!imu_meas_.empty()) {
                        max_time_ns = std::max(imu_meas_.back().time_ns, max_time_ns);
                    }
                    while (!imu_meas_.empty() && imu_meas_.front().time_ns < spline_->maxTimeNs() - spline_->getKnotTimeIntervalNs()) {
                        imu_meas_.pop_front();
                    }

                    // === Milestone 2: Gear System Integration ===
                    if (gear_enabled_ && !imu_meas_.empty()) {
                        std::vector<Eigen::Vector3d> gyros;
                        std::vector<Eigen::Vector3d> accels;
                        for (const auto& imu : imu_meas_) {
                            gyros.push_back(imu.gyro);
                            accels.push_back(imu.accel);
                        }
                        bool needs_resample = gear_system_.updateDecision(gyros, accels);

                        if (needs_resample) {
                            // Upshift: resample control points to new dt
                            gear_spline::StateResampler::resampleInPlace(
                                *spline_,
                                max_time_ns,
                                gear_system_.getCurrentDt(),
                                gear_system_.getActiveN()
                            );
                            ROS_INFO_THROTTLE(1.0, "Gear upshift to %s, dt=%.3fs",
                                gear_spline::GearSystem::modeToString(gear_system_.getCurrentMode()).c_str(),
                                gear_system_.getCurrentDt());
                        }

                        // Precompute blending cache before parallel loops
                        estimator_lio_.precomputeBlendingCache();

                        // Use gear-aware propagation
                        estimator_lio_.propRCPWithGear(max_time_ns, gear_system_.getTargetN());
                    } else {
                        // M1 mode: fixed dt, no gear switching
                        estimator_lio_.propRCP(max_time_ns);
                    }

                    estimator_lio_.updateIEKFLiDARInertial(pt_meas_, &ikdtree, param_.nn_thresh,
                        imu_meas_, gravity_, param_.cov_acc, param_.cov_gyro, param_.coeff_cov);
                }

                // Transform points to world frame
                #pragma omp parallel for num_threads(NUM_OF_THREAD)
                for (size_t i = 0; i < pt_meas_.size(); i++) {
                    PointData& pt_data = pt_meas_[i];
                    Association::pointBodyToWorld(pt_data.time_ns, spline_, pt_data.pt, pt_data.pt_w, pt_data.t_bl, pt_data.q_bl);
                }

                for (size_t i = 0; i < pt_meas_.size(); i++) {
                    PointData& pt_data = pt_meas_[i];
                    pc_world_.points.push_back(pt_data.pt_w);
                    accum_nearest_points_.push_back(pt_data.nearest_points);
                }
                pt_meas_.clear();

                // Publish odometry
                if (spline_->numKnots() > max_spl_knots) {
                    publishOdometry();
                    max_spl_knots = spline_->numKnots();
                }

                // Map update every 100ms
                if (max_time_ns >= t_last_map_upd + int64_t(1e8)) {
                    mapIncremental();
                    publishFrameWorld();
                    lasermapFovSegment();
                    pc_world_.clear();
                    accum_nearest_points_.clear();
                    t_last_map_upd = max_time_ns;
                }
            }

            rate.sleep();
        }
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    ros::NodeHandle& nh_;

    // Subscribers
    ros::Subscriber sub_lidar_;
    ros::Subscriber sub_livox_;
    ros::Subscriber sub_imu_;

    // Publishers
    ros::Publisher pub_odom_;
    ros::Publisher pub_path_;
    ros::Publisher pub_cur_scan_;
    ros::Publisher pub_start_time_;
    ros::Publisher pub_gear_marker_;

    // Gear system (Milestone 2)
    gear_spline::GearSystem gear_system_;
    bool gear_enabled_ = true;  // Set to false to disable gear switching (M1 mode)

    // TF broadcaster
    tf::TransformBroadcaster br_;

    const std::string frame_id_ = "body";
    const std::string odom_id_ = "world";

    // Topics
    std::string topic_lidar_;
    std::string topic_imu_;
    std::string lidar_type_;

    // LiDAR config
    float ds_lm_voxel_;
    pcl::VoxelGrid<pcl::PointXYZINormal> ds_filter_body_;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last_;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last_ds_;
    pcl::PointCloud<pcl::PointXYZINormal> pc_world_;
    int point_filter_num_ = 1;
    int64_t time_offset_ = 0;
    float blind_ = 0.5f;
    int scan_line_ = 6;  // For Livox (default 6 lines for Mid-70/Avia)
    Eigen::Quaterniond q_bl_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t_bl_ = Eigen::Vector3d::Zero();
    double w_pt_ = 0.01;

    // Map management
    std::vector<BoxPointType> cub_needrm_;
    BoxPointType LocalMap_Points_;
    std::vector<Eigen::aligned_vector<pcl::PointXYZINormal>> accum_nearest_points_;
    double cube_len_ = 2000;
    const float MOV_THRESHOLD_ = 1.5f;
    float det_range_ = 100.0;
    bool if_init_map_ = false;

    // LiDAR data buffers
    Eigen::aligned_deque<Eigen::aligned_vector<pcl::PointXYZINormal>> pc_buff_;
    std::deque<int64_t> t_buff_;
    std::mutex mtx_pc_;
    Eigen::aligned_deque<PointData> pt_buff_;
    Eigen::aligned_deque<PointData> pt_meas_;

    // IMU data buffers
    bool if_lidar_only_ = false;
    Eigen::aligned_deque<ImuData> imu_buff_;
    Eigen::aligned_deque<ImuData> imu_meas_;
    std::vector<sensor_msgs::Imu::ConstPtr> imu_int_buff_;
    std::mutex m_buff_;
    bool acc_ratio_ = false;
    Eigen::Vector3d cov_ba_;
    Eigen::Vector3d cov_bg_;
    Eigen::Vector3d gravity_;

    // Filter state
    bool if_init_filter_ = false;
    Estimator<24> estimator_lo_;
    Estimator<30> estimator_lio_;
    SplineState* spline_ = nullptr;
    double cov_P0_ = 0.02;
    double cov_RCP_pos_old_ = 0.02;
    double cov_RCP_ort_old_ = 0.02;
    double cov_RCP_pos_new_ = 0.1;
    double cov_RCP_ort_new_ = 0.1;
    double cov_sys_pos_ = 0.1;
    double cov_sys_ort_ = 0.01;
    Parameters param_;
    int64_t dt_ns_;
    int num_points_upd_;

    // Path for visualization
    nav_msgs::Path path_msg_;

    void readParameters()
    {
        // LiDAR parameters
        topic_lidar_ = CommonUtils::readParam<std::string>(nh_, "topic_lidar");
        lidar_type_ = CommonUtils::readParam<std::string>(nh_, "lidar_type", std::string("Ouster"));

        ds_lm_voxel_ = CommonUtils::readParam<float>(nh_, "ds_lm_voxel", 0.5f);
        float ds_scan_voxel = CommonUtils::readParam<float>(nh_, "ds_scan_voxel", 0.3f);
        ds_filter_body_.setLeafSize(ds_scan_voxel, ds_scan_voxel, ds_scan_voxel);

        param_.nn_thresh = CommonUtils::readParam<double>(nh_, "nn_thresh", 5.0);
        blind_ = CommonUtils::readParam<float>(nh_, "blind", 0.5f);
        scan_line_ = CommonUtils::readParam<int>(nh_, "scan_line", 6);

        // Extrinsic: LiDAR to body
        std::vector<double> q_lb_v = CommonUtils::readParam<std::vector<double>>(nh_, "q_lb",
            std::vector<double>{1.0, 0.0, 0.0, 0.0});
        Eigen::Quaterniond q_lb(q_lb_v[0], q_lb_v[1], q_lb_v[2], q_lb_v[3]);
        std::vector<double> t_lb_v = CommonUtils::readParam<std::vector<double>>(nh_, "t_lb",
            std::vector<double>{0.0, 0.0, 0.0});
        Eigen::Vector3d t_lb(t_lb_v[0], t_lb_v[1], t_lb_v[2]);
        q_bl_ = q_lb.inverse();
        t_bl_ = q_lb.inverse() * (-t_lb);
        w_pt_ = CommonUtils::readParam<double>(nh_, "w_pt", 0.01);

        // IMU/LIO parameters
        if_lidar_only_ = CommonUtils::readParam<bool>(nh_, "if_lidar_only", false);
        if (!if_lidar_only_) {
            topic_imu_ = CommonUtils::readParam<std::string>(nh_, "topic_imu");
            acc_ratio_ = CommonUtils::readParam<bool>(nh_, "acc_ratio", false);

            std::vector<double> bias_acc_var = CommonUtils::readParam<std::vector<double>>(nh_, "cov_ba",
                std::vector<double>{0.0001, 0.0001, 0.0001});
            cov_ba_ << bias_acc_var[0], bias_acc_var[1], bias_acc_var[2];

            std::vector<double> bias_gyro_var = CommonUtils::readParam<std::vector<double>>(nh_, "cov_bg",
                std::vector<double>{0.0001, 0.0001, 0.0001});
            cov_bg_ << bias_gyro_var[0], bias_gyro_var[1], bias_gyro_var[2];

            std::vector<double> acc_var = CommonUtils::readParam<std::vector<double>>(nh_, "cov_acc",
                std::vector<double>{0.1, 0.1, 0.1});
            param_.cov_acc << acc_var[0], acc_var[1], acc_var[2];

            std::vector<double> gyro_var = CommonUtils::readParam<std::vector<double>>(nh_, "cov_gyro",
                std::vector<double>{0.01, 0.01, 0.01});
            param_.cov_gyro << gyro_var[0], gyro_var[1], gyro_var[2];
        }

        // Spline parameters (Milestone 1: fixed dt=0.05s for NORMAL mode)
        int knot_hz = CommonUtils::readParam<int>(nh_, "knot_hz", 20);  // 20Hz = 0.05s
        dt_ns_ = int64_t(1e9) / knot_hz;
        double dt_s = double(dt_ns_) * 1e-9;

        cov_P0_ = CommonUtils::readParam<double>(nh_, "cov_P0", 0.02);
        cov_P0_ *= (dt_s * dt_s);
        cov_RCP_pos_old_ = CommonUtils::readParam<double>(nh_, "cov_RCP_pos_old", 0.02);
        cov_RCP_ort_old_ = CommonUtils::readParam<double>(nh_, "cov_RCP_ort_old", 0.02);
        cov_RCP_pos_new_ = CommonUtils::readParam<double>(nh_, "cov_RCP_pos_new", 0.1);
        cov_RCP_ort_new_ = CommonUtils::readParam<double>(nh_, "cov_RCP_ort_new", 0.1);

        double std_pos = CommonUtils::readParam<double>(nh_, "std_sys_pos", 0.1);
        double std_ort = CommonUtils::readParam<double>(nh_, "std_sys_ort", 0.01);
        cov_sys_pos_ = std_pos * std_pos * dt_s * dt_s;
        cov_sys_ort_ = std_ort * std_ort * dt_s * dt_s;

        param_.coeff_cov = CommonUtils::readParam<double>(nh_, "coeff_cov", 10.0);
        cube_len_ = CommonUtils::readParam<double>(nh_, "cube_len", 2000.0);
        point_filter_num_ = CommonUtils::readParam<int>(nh_, "point_filter_num", 1);
        num_points_upd_ = CommonUtils::readParam<int>(nh_, "num_points_upd", 5000);

        if (if_lidar_only_) {
            estimator_lo_.n_iter = CommonUtils::readParam<int>(nh_, "n_iter", 2);
        } else {
            estimator_lio_.n_iter = CommonUtils::readParam<int>(nh_, "n_iter", 2);
        }

        NUM_MATCH_POINTS = CommonUtils::readParam<int>(nh_, "num_nn", 5);
        NUM_OF_THREAD = CommonUtils::readParam<int>(nh_, "num_threads", 4);

        double lidar_time_offset = CommonUtils::readParam<double>(nh_, "lidar_time_offset", 0.0);
        time_offset_ = int64_t(1e9 * lidar_time_offset);

        // Gear system parameters (Milestone 2)
        gear_enabled_ = CommonUtils::readParam<bool>(nh_, "gear_enabled", true);
        if (!gear_enabled_) {
            ROS_INFO("Gear system DISABLED (M1 mode: fixed dt)");
        } else {
            ROS_INFO("Gear system ENABLED (M2 mode: dynamic dt)");
        }
    }

    void initFilter(int64_t start_t_ns,
                    Eigen::Vector3d t_init = Eigen::Vector3d::Zero(),
                    Eigen::Quaterniond q_init = Eigen::Quaterniond::Identity())
    {
        Eigen::Matrix<double, 24, 24> cov_RCPs = cov_P0_ * Eigen::Matrix<double, 24, 24>::Identity();
        Eigen::Matrix<double, 30, 30> Q = Eigen::Matrix<double, 30, 30>::Zero();

        Eigen::Matrix<double, 6, 6> Q_block_old = Eigen::Matrix<double, 6, 6>::Zero();
        Q_block_old.topLeftCorner<3, 3>() = cov_RCP_pos_old_ * cov_sys_pos_ * Eigen::Matrix3d::Identity();
        Q_block_old.bottomRightCorner<3, 3>() = cov_RCP_ort_old_ * cov_sys_ort_ * Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 6, 6> Q_block_new = Eigen::Matrix<double, 6, 6>::Zero();
        Q_block_new.topLeftCorner<3, 3>() = cov_RCP_pos_new_ * cov_sys_pos_ * Eigen::Matrix3d::Identity();
        Q_block_new.bottomRightCorner<3, 3>() = cov_RCP_ort_new_ * cov_sys_ort_ * Eigen::Matrix3d::Identity();

        Q.topLeftCorner<6, 6>() = Q_block_old;
        Q.block<6, 6>(6, 6) = Q_block_old;
        Q.block<6, 6>(12, 12) = Q_block_old;
        Q.bottomRightCorner<6, 6>() = Q_block_new;

        if (if_lidar_only_) {
            estimator_lo_.setState(dt_ns_, start_t_ns, t_init, q_init, Q.topLeftCorner<24, 24>(), cov_RCPs);
            spline_ = estimator_lo_.getSpline();
        } else {
            Eigen::Matrix<double, 30, 30> cov_x = Eigen::Matrix<double, 30, 30>::Zero();
            cov_x.topLeftCorner<24, 24>() = cov_RCPs;
            cov_x.block<3, 3>(24, 24) = cov_ba_.asDiagonal();
            cov_x.block<3, 3>(27, 27) = cov_bg_.asDiagonal();
            estimator_lio_.setState(dt_ns_, start_t_ns, t_init, q_init, Q, cov_x);
            spline_ = estimator_lio_.getSpline();
        }
    }

    void imuCallback(const sensor_msgs::Imu::ConstPtr& imu_msg)
    {
        m_buff_.lock();
        imu_int_buff_.push_back(imu_msg);
        m_buff_.unlock();
    }

    void ousterLidarCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
    {
        pcl::PointCloud<ouster_ros::Point>::Ptr pc_ouster(new pcl::PointCloud<ouster_ros::Point>());
        pcl::fromROSMsg(*cloud_msg, *pc_ouster);

        size_t plsize = pc_ouster->size();
        if (plsize == 0) return;

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());
        pc_last->reserve(plsize);

        int64_t time_begin = cloud_msg->header.stamp.toNSec() - time_offset_;
        static int64_t last_t_ns = time_begin;
        int64_t max_ofs_ns = 0;

        pcl::PointXYZINormal pt;
        for (unsigned int i = 0; i < plsize; ++i) {
            if (i % point_filter_num_ == 0) {
                pt.x = pc_ouster->points[i].x;
                pt.y = pc_ouster->points[i].y;
                pt.z = pc_ouster->points[i].z;
                pt.intensity = float(pc_ouster->points[i].t) / float(1e6);  // unit: ms
                pt.curvature = pc_ouster->points[i].intensity;

                if (pt.intensity >= 0 &&
                    pt.x * pt.x + pt.y * pt.y + pt.z * pt.z > (blind_ * blind_) &&
                    pc_ouster->points[i].t + time_begin > last_t_ns) {
                    pc_last->points.push_back(pt);
                    int64_t ofs = pc_ouster->points[i].t;
                    max_ofs_ns = std::max(max_ofs_ns, ofs);
                }
            }
        }

        mtx_pc_.lock();
        pc_buff_.push_back(pc_last->points);
        t_buff_.push_back(time_begin);
        mtx_pc_.unlock();

        last_t_ns = time_begin + max_ofs_ns;
    }

    void livoxLidarCallback(const livox_ros_driver::CustomMsg::ConstPtr& livox_msg)
    {
        int plsize = livox_msg->point_num;
        if (plsize == 0) return;

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());
        pc_last->reserve(plsize);

        int64_t time_begin = livox_msg->header.stamp.toNSec() - time_offset_;
        static int64_t last_t_ns = time_begin;
        int64_t max_ofs_ns = 0;

        int valid_point_num = 0;
        pcl::PointXYZINormal pt_pre;
        pt_pre.x = livox_msg->points[0].x;
        pt_pre.y = livox_msg->points[0].y;
        pt_pre.z = livox_msg->points[0].z;

        int N_SCAN_LINES = scan_line_;

        for (int i = 1; i < plsize; ++i) {
            // Check valid points based on tag and line number
            if ((livox_msg->points[i].line < N_SCAN_LINES) &&
                ((livox_msg->points[i].tag & 0x30) == 0x10 || (livox_msg->points[i].tag & 0x30) == 0x00)) {
                valid_point_num++;
                if (valid_point_num % point_filter_num_ == 0) {
                    pcl::PointXYZINormal pt;
                    pt.x = livox_msg->points[i].x;
                    pt.y = livox_msg->points[i].y;
                    pt.z = livox_msg->points[i].z;
                    pt.intensity = float(livox_msg->points[i].offset_time) / float(1e6);  // unit: ms
                    pt.curvature = livox_msg->points[i].reflectivity;

                    // Filter by blind zone and check for duplicates
                    if (pt.intensity >= 0 &&
                        ((std::abs(pt.x - pt_pre.x) > 1e-7) || (std::abs(pt.y - pt_pre.y) > 1e-7) || (std::abs(pt.z - pt_pre.z) > 1e-7)) &&
                        pt.x * pt.x + pt.y * pt.y + pt.z * pt.z > (blind_ * blind_) &&
                        livox_msg->points[i].offset_time + time_begin > last_t_ns) {
                        int64_t ofs = livox_msg->points[i].offset_time;
                        max_ofs_ns = std::max(max_ofs_ns, ofs);
                        pc_last->points.push_back(pt);
                    }
                    pt_pre = pt;
                }
            }
        }

        mtx_pc_.lock();
        pc_buff_.push_back(pc_last->points);
        t_buff_.push_back(time_begin);
        mtx_pc_.unlock();

        last_t_ns = time_begin + max_ofs_ns;
    }

    bool initialization()
    {
        if (if_init_filter_ && if_init_map_) {
            return true;
        }

        if (pt_buff_.empty()) {
            return false;
        }

        int64_t start_t_ns = pt_buff_.front().time_ns;

        if (!if_init_filter_) {
            Eigen::Quaterniond q_WI = Eigen::Quaterniond::Identity();

            if (!if_lidar_only_) {
                Eigen::Vector3d gravity_sum(0, 0, 0);
                m_buff_.lock();
                int buff_size = imu_buff_.size();
                int n_imu = std::min(15, buff_size);
                for (int i = 0; i < n_imu; i++) {
                    gravity_sum += imu_buff_.at(i).accel;
                }
                while (!imu_buff_.empty() && imu_buff_.front().time_ns < start_t_ns) {
                    imu_buff_.pop_front();
                }
                m_buff_.unlock();

                if (n_imu > 0) {
                    gravity_sum /= n_imu;
                    Eigen::Vector3d gravity_ave = gravity_sum.normalized() * 9.81;
                    Eigen::Matrix3d R0 = CommonUtils::g2R(gravity_ave);
                    double yaw = CommonUtils::R2ypr(R0).x();
                    R0 = CommonUtils::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
                    Eigen::Quaterniond q0(R0);
                    q_WI = Quater::positify(q0);
                    gravity_ = q_WI * gravity_ave;
                } else {
                    gravity_ = Eigen::Vector3d(0, 0, -9.81);
                }
            }

            initFilter(start_t_ns, Eigen::Vector3d(0, 0, 0), q_WI);
            if_init_filter_ = true;

            std_msgs::Int64 start_time;
            start_time.data = start_t_ns;
            pub_start_time_.publish(start_time);

            ROS_INFO("Filter initialized at t=%.3fs", start_t_ns * 1e-9);
        }

        if (!if_init_map_) {
            if (ikdtree.Root_Node == nullptr) {
                ikdtree.set_downsample_param(ds_lm_voxel_);
            }

            if (if_lidar_only_) {
                estimator_lo_.propRCP(start_t_ns);
            } else {
                estimator_lio_.propRCP(start_t_ns);
            }

            int feats_down_size = 0;
            for (size_t i = 0; i < pt_buff_.size(); i++) {
                if (pt_buff_[i].time_ns < start_t_ns + int64_t(1e8)) {
                    feats_down_size++;
                } else {
                    break;
                }
            }

            if (feats_down_size < 100) {
                return false;
            }

            pc_world_.clear();
            pc_world_.resize(feats_down_size);

            for (int i = 0; i < feats_down_size; i++) {
                Association::pointBodyToWorld(start_t_ns, spline_, pt_buff_[i].pt,
                    pc_world_.points[i], pt_buff_[i].t_bl, pt_buff_[i].q_bl);
            }

            while (!pt_buff_.empty() && pt_buff_.front().time_ns < start_t_ns + int64_t(1e8)) {
                pt_buff_.pop_front();
            }

            ikdtree.Build(pc_world_.points);
            pc_world_.clear();
            if_init_map_ = true;

            ROS_INFO("Map initialized with %d points", feats_down_size);
        }

        return false;
    }

    bool collectMeasurements()
    {
        if (pt_buff_.empty()) {
            return false;
        }

        int64_t pt_min_time = pt_buff_.front().time_ns;
        int64_t pt_max_time = pt_buff_.back().time_ns;

        if (pt_max_time <= spline_->maxTimeNs() + dt_ns_) {
            return false;
        }

        if (!if_lidar_only_ && (imu_buff_.empty() || imu_buff_.back().time_ns <= spline_->maxTimeNs())) {
            return false;
        }

        int64_t max_time_ns = std::min(spline_->maxTimeNs(), pt_min_time + dt_ns_);

        if (pt_min_time > max_time_ns) {
            if (if_lidar_only_) {
                estimator_lo_.propRCP(pt_min_time);
            } else {
                estimator_lio_.propRCP(pt_min_time);
            }
            max_time_ns = spline_->maxTimeNs();
        }

        if (spline_->numKnots() > 4) {
            max_time_ns = spline_->maxTimeNs();
        }

        int cnt = 0;
        while (!pt_buff_.empty() && pt_buff_.front().time_ns <= max_time_ns && cnt < num_points_upd_) {
            if (spline_->numKnots() < 10 || pt_buff_.front().time_ns >= spline_->maxTimeNs() - dt_ns_) {
                pt_meas_.emplace_back(pt_buff_.front());
            }
            pt_buff_.pop_front();
            cnt++;
        }

        if (!if_lidar_only_) {
            while (!imu_buff_.empty() && imu_buff_.front().time_ns < spline_->minTimeNs()) {
                imu_buff_.pop_front();
            }
            while (!imu_buff_.empty() && imu_buff_.front().time_ns <= max_time_ns) {
                imu_meas_.emplace_back(imu_buff_.front());
                imu_buff_.pop_front();
            }
        }

        return !pt_meas_.empty();
    }

    void publishOdometry()
    {
        int64_t t_ns = spline_->maxTimeNs();
        Eigen::Quaterniond q;
        Eigen::Vector3d pos = spline_->itpPosition(t_ns);
        spline_->itpQuaternion(t_ns, &q);

        // Publish odometry
        nav_msgs::Odometry odom_msg;
        odom_msg.header.stamp.fromNSec(t_ns);
        odom_msg.header.frame_id = odom_id_;
        odom_msg.child_frame_id = frame_id_;
        odom_msg.pose.pose.position.x = pos.x();
        odom_msg.pose.pose.position.y = pos.y();
        odom_msg.pose.pose.position.z = pos.z();
        odom_msg.pose.pose.orientation.w = q.w();
        odom_msg.pose.pose.orientation.x = q.x();
        odom_msg.pose.pose.orientation.y = q.y();
        odom_msg.pose.pose.orientation.z = q.z();
        pub_odom_.publish(odom_msg);

        // Publish TF
        geometry_msgs::TransformStamped tf_msg;
        tf_msg.header.stamp.fromNSec(t_ns);
        tf_msg.header.frame_id = odom_id_;
        tf_msg.child_frame_id = frame_id_;
        tf_msg.transform.translation.x = pos.x();
        tf_msg.transform.translation.y = pos.y();
        tf_msg.transform.translation.z = pos.z();
        tf_msg.transform.rotation.w = q.w();
        tf_msg.transform.rotation.x = q.x();
        tf_msg.transform.rotation.y = q.y();
        tf_msg.transform.rotation.z = q.z();
        br_.sendTransform(tf_msg);

        // Publish path
        geometry_msgs::PoseStamped pose;
        pose.header = odom_msg.header;
        pose.pose = odom_msg.pose.pose;
        path_msg_.header = odom_msg.header;
        path_msg_.poses.push_back(pose);
        pub_path_.publish(path_msg_);

        // Publish Gear Mode Marker (Milestone 2)
        if (gear_enabled_) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = frame_id_;
            marker.header.stamp.fromNSec(t_ns);
            marker.ns = "gear_mode";
            marker.id = 0;
            marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position.x = 0;
            marker.pose.position.y = 0;
            marker.pose.position.z = 2.0;  // Display above robot
            marker.pose.orientation.w = 1.0;
            marker.scale.z = 0.5;

            Eigen::Vector3f color = gear_spline::GearSystem::modeToColor(gear_system_.getCurrentMode());
            marker.color.r = color.x();
            marker.color.g = color.y();
            marker.color.b = color.z();
            marker.color.a = 1.0;

            std::stringstream ss;
            ss << gear_spline::GearSystem::modeToString(gear_system_.getCurrentMode())
               << " (E=" << std::fixed << std::setprecision(2) << gear_system_.getLastEnergy() << ")";
            marker.text = ss.str();
            marker.lifetime = ros::Duration(0.2);
            pub_gear_marker_.publish(marker);
        }
    }

    void publishFrameWorld()
    {
        int size = pc_world_.points.size();
        pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudWorld(new pcl::PointCloud<pcl::PointXYZI>(size, 1));
        for (int i = 0; i < size; i++) {
            laserCloudWorld->points[i].x = pc_world_.points[i].x;
            laserCloudWorld->points[i].y = pc_world_.points[i].y;
            laserCloudWorld->points[i].z = pc_world_.points[i].z;
            laserCloudWorld->points[i].intensity = pc_world_.points[i].curvature;
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp.fromNSec(spline_->maxTimeNs());
        laserCloudmsg.header.frame_id = odom_id_;
        pub_cur_scan_.publish(laserCloudmsg);
    }

    Eigen::Vector3d getPositionLiDAR(int64_t t_ns, const Eigen::Vector3d& t_bl)
    {
        if (if_lidar_only_) {
            estimator_lo_.propRCP(t_ns);
        } else {
            estimator_lio_.propRCP(t_ns);
        }
        Eigen::Quaterniond orient_interp;
        Eigen::Vector3d t_interp = spline_->itpPosition(t_ns);
        spline_->itpQuaternion(t_ns, &orient_interp);
        return orient_interp * t_bl + t_interp;
    }

    void lasermapFovSegment()
    {
        static bool Localmap_Initialized = false;
        cub_needrm_.shrink_to_fit();

        Eigen::Vector3d pos_lidar = getPositionLiDAR(spline_->maxTimeNs(), t_bl_);

        if (!Localmap_Initialized) {
            for (int i = 0; i < 3; i++) {
                LocalMap_Points_.vertex_min[i] = pos_lidar(i) - cube_len_ / 2.0;
                LocalMap_Points_.vertex_max[i] = pos_lidar(i) + cube_len_ / 2.0;
            }
            Localmap_Initialized = true;
            return;
        }

        float dist_to_map_edge[3][2];
        bool need_move = false;
        for (int i = 0; i < 3; i++) {
            dist_to_map_edge[i][0] = fabs(pos_lidar(i) - LocalMap_Points_.vertex_min[i]);
            dist_to_map_edge[i][1] = fabs(pos_lidar(i) - LocalMap_Points_.vertex_max[i]);
            if (dist_to_map_edge[i][0] <= MOV_THRESHOLD_ * det_range_ ||
                dist_to_map_edge[i][1] <= MOV_THRESHOLD_ * det_range_) {
                need_move = true;
            }
        }

        if (!need_move) return;

        BoxPointType New_LocalMap_Points, tmp_boxpoints;
        New_LocalMap_Points = LocalMap_Points_;
        float mov_dist = std::max((cube_len_ - 2.0 * MOV_THRESHOLD_ * det_range_) * 0.5 * 0.9,
                                  double(det_range_ * (MOV_THRESHOLD_ - 1)));

        for (int i = 0; i < 3; i++) {
            tmp_boxpoints = LocalMap_Points_;
            if (dist_to_map_edge[i][0] <= MOV_THRESHOLD_ * det_range_) {
                New_LocalMap_Points.vertex_max[i] -= mov_dist;
                New_LocalMap_Points.vertex_min[i] -= mov_dist;
                tmp_boxpoints.vertex_min[i] = LocalMap_Points_.vertex_max[i] - mov_dist;
                cub_needrm_.emplace_back(tmp_boxpoints);
            } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD_ * det_range_) {
                New_LocalMap_Points.vertex_max[i] += mov_dist;
                New_LocalMap_Points.vertex_min[i] += mov_dist;
                tmp_boxpoints.vertex_max[i] = LocalMap_Points_.vertex_min[i] + mov_dist;
                cub_needrm_.emplace_back(tmp_boxpoints);
            }
        }
        LocalMap_Points_ = New_LocalMap_Points;

        if (!cub_needrm_.empty()) {
            ikdtree.Delete_Point_Boxes(cub_needrm_);
        }
    }

    void mapIncremental()
    {
        Eigen::aligned_vector<pcl::PointXYZINormal> PointToAdd;
        Eigen::aligned_vector<pcl::PointXYZINormal> PointNoNeedDownsample;
        int feats_down_size = pc_world_.points.size();
        PointToAdd.reserve(feats_down_size);
        PointNoNeedDownsample.reserve(feats_down_size);

        for (int i = 0; i < feats_down_size; i++) {
            const pcl::PointXYZINormal& point = pc_world_.points[i];
            if (!accum_nearest_points_[i].empty()) {
                const Eigen::aligned_vector<pcl::PointXYZINormal>& points_near = accum_nearest_points_[i];
                bool need_add = true;
                pcl::PointXYZINormal mid_point;

                mid_point.x = floor(point.x / ds_lm_voxel_) * ds_lm_voxel_ + 0.5 * ds_lm_voxel_;
                mid_point.y = floor(point.y / ds_lm_voxel_) * ds_lm_voxel_ + 0.5 * ds_lm_voxel_;
                mid_point.z = floor(point.z / ds_lm_voxel_) * ds_lm_voxel_ + 0.5 * ds_lm_voxel_;

                if (fabs(points_near[0].x - mid_point.x) > 0.866 * ds_lm_voxel_ ||
                    fabs(points_near[0].y - mid_point.y) > 0.866 * ds_lm_voxel_ ||
                    fabs(points_near[0].z - mid_point.z) > 0.866 * ds_lm_voxel_) {
                    PointNoNeedDownsample.emplace_back(pc_world_.points[i]);
                    continue;
                }

                for (size_t readd_i = 0; readd_i < points_near.size(); readd_i++) {
                    if (fabs(points_near[readd_i].x - mid_point.x) < 0.5 * ds_lm_voxel_ &&
                        fabs(points_near[readd_i].y - mid_point.y) < 0.5 * ds_lm_voxel_ &&
                        fabs(points_near[readd_i].z - mid_point.z) < 0.5 * ds_lm_voxel_) {
                        need_add = false;
                        break;
                    }
                }
                if (need_add) PointToAdd.emplace_back(point);
            } else {
                PointNoNeedDownsample.emplace_back(point);
            }
        }

        ikdtree.Add_Points(PointToAdd, true);
        ikdtree.Add_Points(PointNoNeedDownsample, false);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "gear_spline_lio_node");
    ros::NodeHandle nh("~");

    GearSplineLIO lio(nh);

    ROS_INFO("Gear-Spline LIO node started!");

    std::thread processing_thread(&GearSplineLIO::processData, &lio);

    ros::Rate rate(200);
    while (ros::ok()) {
        ros::spinOnce();
        rate.sleep();
    }

    processing_thread.join();

    return 0;
}
