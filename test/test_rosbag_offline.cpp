/**
 * @file test_rosbag_offline.cpp
 * @brief Offline rosbag test for Gear-Spline LIO
 *
 * This program preloads LiDAR and IMU data from rosbag,
 * then runs the IESKF estimation pipeline directly.
 *
 * Easier to debug than ROS callback-based approach.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <chrono>
#include <algorithm>
#include <iomanip>

// ROS for rosbag API
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver/CustomMsg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Gear-Spline core headers
#include "gear_spline/SplineState.hpp"
#include "gear_spline/Estimator.hpp"
#include "gear_spline/Association.hpp"
#include "gear_spline/ikd-Tree/ikd_Tree.h"
#include "gear_spline/utils/common_utils.h"
#include "gear_spline/utils/math_tools.h"

// Global variables required by Estimator/Association
int NUM_OF_THREAD = 4;
int NUM_MATCH_POINTS = 5;

using namespace std;

// ========= Data Structures =========
struct ImuMeasurement {
    double timestamp;
    Eigen::Vector3d acc;
    Eigen::Vector3d gyro;
};

struct LidarFrame {
    double timestamp;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud;
    vector<double> point_times;  // Per-point timestamps (relative to frame start)
};

// ========= Logger =========
class Logger {
public:
    enum Level { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };

    static void log(Level level, const string& msg) {
        const char* level_str[] = {"DEBUG", "INFO", "WARN", "ERROR"};
        auto now = chrono::system_clock::now();
        auto time_t = chrono::system_clock::to_time_t(now);
        cout << "[" << put_time(localtime(&time_t), "%H:%M:%S") << "] "
             << "[" << level_str[level] << "] " << msg << endl;
    }

    static void debug(const string& msg) { log(DEBUG, msg); }
    static void info(const string& msg) { log(INFO, msg); }
    static void warn(const string& msg) { log(WARN, msg); }
    static void error(const string& msg) { log(ERROR, msg); }
};

// ========= Rosbag Data Loader =========
class RosbagLoader {
public:
    string bag_path_;
    string lidar_topic_;
    string imu_topic_;
    string lidar_type_;  // "Livox" or "Ouster"
    int scan_line_ = 6;
    double blind_radius_ = 0.5;
    int point_filter_num_ = 3;

    vector<ImuMeasurement> imu_data_;
    vector<LidarFrame> lidar_data_;

    bool load() {
        Logger::info("Opening rosbag: " + bag_path_);

        try {
            rosbag::Bag bag;
            bag.open(bag_path_, rosbag::bagmode::Read);

            // Read topics
            vector<string> topics = {lidar_topic_, imu_topic_};
            rosbag::View view(bag, rosbag::TopicQuery(topics));

            Logger::info("Reading messages from topics: " + lidar_topic_ + ", " + imu_topic_);

            int imu_count = 0, lidar_count = 0;

            for (const rosbag::MessageInstance& m : view) {
                if (m.getTopic() == imu_topic_) {
                    sensor_msgs::Imu::ConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
                    if (imu_msg) {
                        ImuMeasurement imu;
                        imu.timestamp = imu_msg->header.stamp.toSec();
                        imu.acc = Eigen::Vector3d(
                            imu_msg->linear_acceleration.x,
                            imu_msg->linear_acceleration.y,
                            imu_msg->linear_acceleration.z
                        );
                        imu.gyro = Eigen::Vector3d(
                            imu_msg->angular_velocity.x,
                            imu_msg->angular_velocity.y,
                            imu_msg->angular_velocity.z
                        );
                        imu_data_.push_back(imu);
                        imu_count++;
                    }
                }
                else if (m.getTopic() == lidar_topic_) {
                    if (lidar_type_ == "Livox" || lidar_type_ == "Mid70Avia") {
                        livox_ros_driver::CustomMsg::ConstPtr livox_msg =
                            m.instantiate<livox_ros_driver::CustomMsg>();
                        if (livox_msg) {
                            processLivoxMsg(livox_msg);
                            lidar_count++;
                        }
                    } else {
                        sensor_msgs::PointCloud2::ConstPtr cloud_msg =
                            m.instantiate<sensor_msgs::PointCloud2>();
                        if (cloud_msg) {
                            processPointCloud2Msg(cloud_msg);
                            lidar_count++;
                        }
                    }
                }
            }

            bag.close();

            Logger::info("Loaded " + to_string(imu_count) + " IMU messages");
            Logger::info("Loaded " + to_string(lidar_count) + " LiDAR frames");

            // Sort by timestamp
            sort(imu_data_.begin(), imu_data_.end(),
                 [](const ImuMeasurement& a, const ImuMeasurement& b) {
                     return a.timestamp < b.timestamp;
                 });
            sort(lidar_data_.begin(), lidar_data_.end(),
                 [](const LidarFrame& a, const LidarFrame& b) {
                     return a.timestamp < b.timestamp;
                 });

            if (!imu_data_.empty() && !lidar_data_.empty()) {
                Logger::info("Time range: IMU [" + to_string(imu_data_.front().timestamp) +
                            " - " + to_string(imu_data_.back().timestamp) + "]");
                Logger::info("Time range: LiDAR [" + to_string(lidar_data_.front().timestamp) +
                            " - " + to_string(lidar_data_.back().timestamp) + "]");
            }

            return true;
        }
        catch (const exception& e) {
            Logger::error("Failed to load rosbag: " + string(e.what()));
            return false;
        }
    }

private:
    void processLivoxMsg(const livox_ros_driver::CustomMsg::ConstPtr& msg) {
        LidarFrame frame;
        frame.timestamp = msg->header.stamp.toSec();
        frame.cloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>());

        int plsize = msg->point_num;
        static bool first_msg = true;
        if (first_msg && plsize > 0) {
            Logger::debug("First Livox msg: " + to_string(plsize) + " points, tag=" +
                         to_string(msg->points[0].tag) + ", line=" + to_string(msg->points[0].line));
            first_msg = false;
        }

        for (int i = 0; i < plsize; i++) {
            // Relaxed filtering for Livox - skip only obvious noise
            // Original: (msg->points[i].tag & 0x30) != 0x10, line < 6
            // Keep all valid points for now
            if (i % point_filter_num_ != 0) continue;

            double x = msg->points[i].x;
            double y = msg->points[i].y;
            double z = msg->points[i].z;
            double dist = sqrt(x*x + y*y + z*z);

            // Filter by blind radius
            if (dist < blind_radius_ || dist > 100.0) continue;
            // Skip invalid points (NaN check)
            if (x != x || y != y || z != z) continue;

            pcl::PointXYZINormal pt;
            pt.x = x;
            pt.y = y;
            pt.z = z;
            pt.intensity = msg->points[i].reflectivity;
            pt.curvature = (msg->points[i].offset_time / 1e9);  // Store relative time in curvature

            frame.cloud->push_back(pt);
            frame.point_times.push_back(msg->points[i].offset_time / 1e9);
        }

        if (frame.cloud->size() > 10) {
            lidar_data_.push_back(frame);
        }
    }

    void processPointCloud2Msg(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        LidarFrame frame;
        // Ouster: lidar_timestamp_end = true, scan timestamp is at the END
        // Subtract scan duration (~100ms) to get scan START time
        frame.timestamp = msg->header.stamp.toSec() - 0.1003;
        frame.cloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>());

        // Use Ouster point type
        pcl::PointCloud<ouster_ros::Point> raw_cloud;
        pcl::fromROSMsg(*msg, raw_cloud);

        for (size_t i = 0; i < raw_cloud.size(); i++) {
            if (i % point_filter_num_ != 0) continue;

            const auto& pt_raw = raw_cloud.points[i];
            double dist = sqrt(pt_raw.x*pt_raw.x + pt_raw.y*pt_raw.y + pt_raw.z*pt_raw.z);

            // Distance filtering (Coco-LIC viral: 1.0 ~ 200.0m)
            if (dist < 1.0 || dist > 200.0) continue;

            // Time offset filtering (should be < 110ms for 10Hz scan)
            if (pt_raw.t > 0.11e9) continue;

            pcl::PointXYZINormal pt;
            pt.x = pt_raw.x;
            pt.y = pt_raw.y;
            pt.z = pt_raw.z;
            pt.intensity = pt_raw.intensity;
            pt.curvature = pt_raw.t / 1e9;  // Convert ns to seconds

            frame.cloud->push_back(pt);
            frame.point_times.push_back(pt_raw.t / 1e9);
        }

        if (frame.cloud->size() > 10) {
            lidar_data_.push_back(frame);
        }
    }
};

// ========= Offline IESKF Runner =========
class OfflineIESKFRunner {
public:
    // Parameters
    double ds_lm_voxel_ = 0.5;
    double ds_scan_voxel_ = 0.3;
    double nn_thresh_ = 5.0;
    int num_points_upd_ = 5000;
    int num_nn_ = 5;
    double w_pt_ = 0.01;
    Eigen::Vector3d cov_ba_{0.0001, 0.0001, 0.0001};
    Eigen::Vector3d cov_bg_{0.0001, 0.0001, 0.0001};
    Eigen::Vector3d cov_acc_{0.1, 0.1, 0.1};
    Eigen::Vector3d cov_gyro_{0.01, 0.01, 0.01};
    int knot_hz_ = 20;
    int n_iter_ = 2;
    double cube_len_ = 2000.0;

    // Extrinsics (Coco-LIC viral config: LiDAR to IMU)
    Eigen::Quaterniond q_lb_{1, 0, 0, 0};  // Identity rotation
    Eigen::Vector3d t_lb_{-0.050, 0.0, 0.055};  // Trans: [-0.050, 0.0, 0.055]

    // State
    SplineState spline_;
    Estimator<30>* estimator_ = nullptr;  // 30 = 24 (4 CPs * 6) + 6 (bias)
    KD_TREE<pcl::PointXYZINormal>* ikd_tree_ = nullptr;

    // Type aliases for ikd-tree
    typedef KD_TREE<pcl::PointXYZINormal>::PointVector PointVector;

    bool initialized_ = false;
    int frame_count_ = 0;

    // Gravity (Coco-LIC convention: [0, 0, +9.80] pointing up)
    Eigen::Vector3d gravity_{0, 0, 9.80};

    // IMU integration state
    Eigen::Vector3d current_position_{0, 0, 0};
    Eigen::Vector3d current_velocity_{0, 0, 0};
    Eigen::Quaterniond current_orientation_{1, 0, 0, 0};

    // Output trajectory
    vector<pair<double, Eigen::Matrix4d>> trajectory_;

    OfflineIESKFRunner() {
        ikd_tree_ = new KD_TREE<pcl::PointXYZINormal>(0.3, 0.6, 0.2);
    }

    ~OfflineIESKFRunner() {
        delete ikd_tree_;
        if (estimator_) delete estimator_;
    }

    void run(const vector<ImuMeasurement>& imu_data,
             const vector<LidarFrame>& lidar_data) {

        Logger::info("Starting IESKF processing...");
        Logger::info("Total frames to process: " + to_string(lidar_data.size()));

        double dt_ns = 1e9 / knot_hz_;  // Knot interval in nanoseconds

        size_t imu_idx = 0;

        for (size_t frame_idx = 0; frame_idx < lidar_data.size(); frame_idx++) {
            const LidarFrame& frame = lidar_data[frame_idx];

            Logger::debug("Processing frame " + to_string(frame_idx) +
                         " at t=" + to_string(frame.timestamp) +
                         " with " + to_string(frame.cloud->size()) + " points");

            // Collect IMU data for this frame
            vector<ImuMeasurement> frame_imu;
            while (imu_idx < imu_data.size() &&
                   imu_data[imu_idx].timestamp <= frame.timestamp) {
                frame_imu.push_back(imu_data[imu_idx]);
                imu_idx++;
            }

            Logger::debug("  IMU samples for this frame: " + to_string(frame_imu.size()));

            // Initialize if needed
            if (!initialized_) {
                if (!initialize(frame, frame_imu)) {
                    Logger::warn("  Initialization failed, skipping frame");
                    continue;
                }
                Logger::info("  Initialization successful!");
                initialized_ = true;
                continue;
            }

            // Process frame
            processFrame(frame, frame_imu, dt_ns);

            frame_count_++;

            // Log progress every 10 frames
            if (frame_count_ % 10 == 0) {
                Eigen::Vector3d pos = spline_.itpPosition(
                    static_cast<int64_t>(frame.timestamp * 1e9), nullptr);
                Logger::info("Frame " + to_string(frame_count_) +
                            " position: [" + to_string(pos.x()) + ", " +
                            to_string(pos.y()) + ", " + to_string(pos.z()) + "]");
            }
        }

        Logger::info("Processing complete! Processed " + to_string(frame_count_) + " frames");
    }

private:
    bool initialize(const LidarFrame& frame, const vector<ImuMeasurement>& imu) {
        Logger::debug("  Initializing spline and estimator...");

        if (imu.size() < 10) {
            Logger::warn("  Not enough IMU samples for initialization (" +
                        to_string(imu.size()) + ")");
            return false;
        }

        // Compute average gravity direction from IMU accelerometer
        Eigen::Vector3d acc_mean = Eigen::Vector3d::Zero();
        for (const auto& m : imu) {
            acc_mean += m.acc;
        }
        acc_mean /= imu.size();

        Logger::debug("  Mean accelerometer: [" + to_string(acc_mean.x()) + ", " +
                     to_string(acc_mean.y()) + ", " + to_string(acc_mean.z()) + "]");

        // Estimate initial orientation using gravity alignment (g2R)
        // g2R computes R such that R * acc_normalized = [0, 0, 1]
        Eigen::Matrix3d R_init = CommonUtils::g2R(acc_mean);
        current_orientation_ = Eigen::Quaterniond(R_init);
        current_orientation_.normalize();

        Logger::debug("  Initial orientation (wxyz): [" +
                     to_string(current_orientation_.w()) + ", " +
                     to_string(current_orientation_.x()) + ", " +
                     to_string(current_orientation_.y()) + ", " +
                     to_string(current_orientation_.z()) + "]");

        // World frame gravity: pointing down in Z (negative)
        // When stationary, accelerometer reads reaction force = -gravity in body frame
        // After rotation to world: R * acc_body should give [0, 0, +g] for upward reaction
        gravity_ = Eigen::Vector3d(0, 0, acc_mean.norm());

        Logger::debug("  Gravity compensation: [0, 0, " + to_string(gravity_.z()) + "]");

        // Initialize spline at origin with identity rotation
        int64_t start_time_ns = static_cast<int64_t>(frame.timestamp * 1e9);
        int64_t dt_ns = static_cast<int64_t>(1e9 / knot_hz_);

        Logger::debug("  Spline start time: " + to_string(start_time_ns) + " ns");
        Logger::debug("  Knot interval: " + to_string(dt_ns) + " ns");

        // Create initial control points (4 knots) with gravity-aligned orientation
        Eigen::Vector3d initial_pos = Eigen::Vector3d::Zero();

        // Convert initial orientation to axis-angle for spline
        Eigen::AngleAxisd aa_init(current_orientation_);
        Eigen::Vector3d initial_ort_delta = aa_init.angle() * aa_init.axis();

        Logger::debug("  Initial orientation delta: [" +
                     to_string(initial_ort_delta.x()) + ", " +
                     to_string(initial_ort_delta.y()) + ", " +
                     to_string(initial_ort_delta.z()) + "]");

        // Initialize spline with: dt_ns, num_knots=0, start_time_ns
        spline_.init(dt_ns, 0, start_time_ns);
        for (int i = 0; i < 4; i++) {
            spline_.addOneStateKnot(initial_pos, initial_ort_delta);
        }

        Logger::debug("  Spline initialized with " + to_string(spline_.numKnots()) + " knots");
        Logger::debug("  Spline time range: [" + to_string(spline_.minTimeNs()) +
                     ", " + to_string(spline_.maxTimeNs()) + "] ns");

        // Initialize ikd-tree with first frame (transform to world frame)
        if (frame.cloud->size() > 100) {
            Logger::debug("  Building ikd-tree with " + to_string(frame.cloud->size()) + " points");

            // Get initial pose from spline (identity at origin)
            int64_t init_time_ns = static_cast<int64_t>(frame.timestamp * 1e9);
            Eigen::Quaterniond q_init;
            spline_.itpQuaternion(init_time_ns, &q_init, nullptr, nullptr, nullptr);
            Eigen::Vector3d p_init = spline_.itpPosition(init_time_ns, nullptr);

            // Transform points to world frame
            PointVector init_points;
            for (const auto& pt : *frame.cloud) {
                Eigen::Vector3d pt_b(pt.x, pt.y, pt.z);
                Eigen::Vector3d pt_w = q_init * (q_lb_ * pt_b + t_lb_) + p_init;

                pcl::PointXYZINormal pt_world;
                pt_world.x = pt_w.x();
                pt_world.y = pt_w.y();
                pt_world.z = pt_w.z();
                pt_world.intensity = pt.intensity;
                init_points.push_back(pt_world);
            }

            ikd_tree_->Build(init_points);
            Logger::debug("  ikd-tree built with world-frame points");
        }

        // Create and initialize estimator
        if (estimator_) delete estimator_;
        estimator_ = new Estimator<30>();

        // Initialize estimator state
        Eigen::Matrix<double, 30, 30> Q = Eigen::Matrix<double, 30, 30>::Identity() * 0.001;
        Eigen::Matrix<double, 30, 30> P = Eigen::Matrix<double, 30, 30>::Identity() * 0.1;

        // Set smaller covariance for bias terms
        for (int i = 24; i < 30; i++) {
            Q(i, i) = 0.0001;
            P(i, i) = 0.01;
        }

        estimator_->setState(dt_ns, start_time_ns, initial_pos, Eigen::Quaterniond::Identity(), Q, P);
        estimator_->n_iter = n_iter_;
        Logger::debug("  Estimator initialized with state size 30");

        return true;
    }

    void processFrame(const LidarFrame& frame,
                      const vector<ImuMeasurement>& imu,
                      double dt_ns) {

        int64_t frame_time_ns = static_cast<int64_t>(frame.timestamp * 1e9);

        // Integrate IMU for position and orientation prediction
        // Gravity is subtracted in world frame after rotating the accelerometer reading
        if (!imu.empty() && imu.size() >= 2) {
            for (size_t i = 1; i < imu.size(); i++) {
                double dt = imu[i].timestamp - imu[i-1].timestamp;
                if (dt > 0 && dt < 0.1) {
                    // Integrate orientation using gyro first
                    Eigen::Vector3d omega = imu[i].gyro;
                    double angle = omega.norm() * dt;
                    if (angle > 1e-8) {
                        Eigen::Quaterniond dq(Eigen::AngleAxisd(angle, omega.normalized()));
                        current_orientation_ = current_orientation_ * dq;
                        current_orientation_.normalize();
                    }

                    // Rotate accelerometer to world frame
                    // acc_body is specific force = linear_acceleration + gravity_in_body_frame
                    // After rotating to world: acc_world = R * acc_body
                    // To get linear acceleration: linear_acc = R * acc_body - gravity_world
                    // where gravity_world = [0, 0, -9.8] points downward
                    Eigen::Vector3d acc_world = current_orientation_ * imu[i].acc;

                    // Subtract gravity (pointing down in world frame)
                    // When stationary and level, acc_body ≈ [0, 0, 9.8] (reaction to gravity)
                    // After rotation (if level): acc_world ≈ [0, 0, 9.8]
                    // Linear acceleration = [0, 0, 9.8] - [0, 0, 9.8] = [0, 0, 0] ✓
                    Eigen::Vector3d gravity_world(0, 0, gravity_.z()); // gravity_ was set to [0, 0, |acc_mean|]
                    Eigen::Vector3d linear_acc = acc_world - gravity_world;

                    // Integrate velocity and position
                    current_velocity_ += linear_acc * dt;
                    current_position_ += current_velocity_ * dt + 0.5 * linear_acc * dt * dt;
                }
            }
        }

        // Step 1: Propagate spline to cover ALL point times (not just frame start)
        Logger::debug("  Step 1: Propagating spline...");

        // Calculate max point time = frame_time + max(curvature)
        int64_t max_pt_time_ns = frame_time_ns;
        for (const auto& pt : *frame.cloud) {
            int64_t pt_time = frame_time_ns + static_cast<int64_t>(pt.curvature * 1e9);
            max_pt_time_ns = std::max(max_pt_time_ns, pt_time);
        }
        // Add margin for B-spline interpolation (need 4 control points)
        int64_t target_time_ns = max_pt_time_ns + static_cast<int64_t>(dt_ns);

        while (spline_.maxTimeNs() < target_time_ns) {
            // Use IMU-integrated position for new control point
            Eigen::Vector3d new_pos = current_position_;

            // Use current gyro-integrated orientation for rotation
            Eigen::AngleAxisd aa(current_orientation_);
            Eigen::Vector3d new_ort_delta = aa.angle() * aa.axis();

            spline_.addOneStateKnot(new_pos, new_ort_delta);

            Logger::debug("    Added knot at " + to_string(spline_.maxTimeNs()) + " ns");
        }

        // Step 2: Prepare point data
        Logger::debug("  Step 2: Preparing " + to_string(frame.cloud->size()) + " points...");
        Eigen::aligned_deque<PointData> pt_meas;

        int valid_points = 0;
        for (size_t i = 0; i < frame.cloud->size() && valid_points < num_points_upd_; i++) {
            const auto& pt = frame.cloud->points[i];

            PointData pd;
            pd.pt_b = Eigen::Vector3d(pt.x, pt.y, pt.z);
            pd.time_ns = frame_time_ns + static_cast<int64_t>(pt.curvature * 1e9);
            pd.q_bl = q_lb_;
            pd.t_bl = t_lb_;
            pd.if_valid = true;

            pt_meas.push_back(pd);
            valid_points++;
        }

        Logger::debug("    Prepared " + to_string(pt_meas.size()) + " valid points");

        // Step 3: Find correspondences
        Logger::debug("  Step 3: Finding correspondences...");
        Logger::debug("    Spline range: [" + to_string(spline_.minTimeNs()) +
                     ", " + to_string(spline_.maxTimeNs()) + "]");

        // Debug: Check first point's timestamp
        if (!pt_meas.empty()) {
            int64_t first_pt_time = pt_meas[0].time_ns;
            int64_t last_pt_time = pt_meas.back().time_ns;
            Logger::debug("    Point time range: [" + to_string(first_pt_time) +
                         ", " + to_string(last_pt_time) + "]");

            bool in_range = (first_pt_time >= spline_.minTimeNs() &&
                            last_pt_time <= spline_.maxTimeNs());
            Logger::debug("    Points in spline range: " + string(in_range ? "YES" : "NO"));
        }

        int num_corr = 0;
        int out_of_range = 0;
        int no_neighbors = 0;
        int bad_plane = 0;
        bool first_debug = true;

        for (auto& pd : pt_meas) {
            if (!pd.if_valid) continue;

            // Check if point is in valid spline range
            if (pd.time_ns < spline_.minTimeNs() || pd.time_ns > spline_.maxTimeNs()) {
                pd.if_valid = false;
                out_of_range++;
                continue;
            }

            // Transform point to world frame
            Eigen::Quaterniond q;
            spline_.itpQuaternion(pd.time_ns, &q, nullptr, nullptr, nullptr);
            Eigen::Vector3d p = spline_.itpPosition(pd.time_ns, nullptr);

            Eigen::Vector3d pt_w = q * (pd.q_bl * pd.pt_b + pd.t_bl) + p;

            // Debug first point
            if (first_debug) {
                Logger::debug("    First point: body=(" + to_string(pd.pt_b.x()) + ", " +
                             to_string(pd.pt_b.y()) + ", " + to_string(pd.pt_b.z()) + ")");
                Logger::debug("    First point: world=(" + to_string(pt_w.x()) + ", " +
                             to_string(pt_w.y()) + ", " + to_string(pt_w.z()) + ")");
                Logger::debug("    Spline pos=(" + to_string(p.x()) + ", " +
                             to_string(p.y()) + ", " + to_string(p.z()) + ")");
                first_debug = false;
            }

            // Find nearest neighbors
            pcl::PointXYZINormal search_pt;
            search_pt.x = pt_w.x();
            search_pt.y = pt_w.y();
            search_pt.z = pt_w.z();

            PointVector nearest;
            vector<float> distances;

            ikd_tree_->Nearest_Search(search_pt, num_nn_, nearest, distances);

            if (nearest.size() >= 3) {
                // Fit plane to neighbors
                Eigen::Vector3d center = Eigen::Vector3d::Zero();
                for (const auto& n : nearest) {
                    center += Eigen::Vector3d(n.x, n.y, n.z);
                }
                center /= nearest.size();

                // Compute covariance
                Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
                for (const auto& n : nearest) {
                    Eigen::Vector3d d = Eigen::Vector3d(n.x, n.y, n.z) - center;
                    cov += d * d.transpose();
                }
                cov /= nearest.size();

                // SVD to find normal
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
                Eigen::Vector3d normal = solver.eigenvectors().col(0);

                // Check if plane is valid (smallest eigenvalue should be small)
                if (solver.eigenvalues()(0) < 0.1 * solver.eigenvalues()(2)) {
                    pd.normvec = normal;
                    pd.dist = -normal.dot(center);
                    pd.if_valid = true;
                    num_corr++;
                } else {
                    pd.if_valid = false;
                    bad_plane++;
                }
            } else {
                pd.if_valid = false;
                no_neighbors++;
            }
        }

        Logger::debug("    Found " + to_string(num_corr) + " valid correspondences");
        Logger::debug("    Failures: out_of_range=" + to_string(out_of_range) +
                     ", no_neighbors=" + to_string(no_neighbors) +
                     ", bad_plane=" + to_string(bad_plane));

        // Step 4: IESKF Update with NaN detection
        Logger::debug("  Step 4: IESKF Update...");

        // Require minimum correspondences for stable update
        const int MIN_CORRESPONDENCES = 50;
        if (num_corr >= MIN_CORRESPONDENCES) {
            // Check spline state before update
            Eigen::Vector3d pos_before = spline_.itpPosition(frame_time_ns, nullptr);
            if (!pos_before.allFinite()) {
                Logger::warn("    NaN detected BEFORE IESKF - skipping update");
            } else {
                // The Estimator has a fixed state size for 4 control points (XSIZE=30)
                // So we can't add more knots to it - we sync only the last 4 RCPs

                SplineState* est_spline = estimator_->getSpline();

                // FIX: First propagate estimator's spline to cover ALL point times
                // This ensures Association::findCorresp inside updateIEKFLiDAR accepts all points
                // Use target_time_ns (which covers max_pt_time_ns + dt_ns margin)
                estimator_->propRCP(target_time_ns);

                // Now sync the last 4 control points from local spline to estimator
                Eigen::Matrix<double, 24, 1> local_rcps = spline_.getRCPs();
                est_spline->updateRCPs(local_rcps);

                Logger::debug("    Estimator spline: " + to_string(est_spline->numKnots()) + " knots");
                Logger::debug("    Est range: [" + to_string(est_spline->minTimeNs()) +
                             ", " + to_string(est_spline->maxTimeNs()) + "]");

                // Run IESKF LiDAR update
                double pt_thresh = 0.5;
                double cov_thresh = 100.0;

                // Debug: RCPs before
                Eigen::Matrix<double, 24, 1> before_rcps = est_spline->getRCPs();

                estimator_->updateIEKFLiDAR(pt_meas, ikd_tree_, pt_thresh, cov_thresh);

                // Get updated RCPs from estimator
                Eigen::Matrix<double, 24, 1> updated_rcps = est_spline->getRCPs();

                // Debug: Show change in RCPs
                double rcp_change = (updated_rcps - before_rcps).norm();
                if (frame_count_ % 100 == 0) {
                    Logger::debug("    RCP change: " + to_string(rcp_change));
                    Logger::debug("    Last CP pos: [" + to_string(updated_rcps(12)) + ", " +
                                 to_string(updated_rcps(13)) + ", " + to_string(updated_rcps(14)) + "]");
                }

                // Check for NaN before applying
                if (updated_rcps.allFinite()) {
                    spline_.updateRCPs(updated_rcps);
                    Logger::debug("    IESKF update completed");
                } else {
                    Logger::warn("    NaN in updated RCPs - discarding update");
                }
            }
        } else {
            Logger::warn("    Skipped IESKF: " + to_string(num_corr) + " correspondences < " +
                        to_string(MIN_CORRESPONDENCES) + " required");
        }

        // Step 5: Update map
        Logger::debug("  Step 5: Updating map...");
        if (frame_count_ % 5 == 0) {  // Add to map every 5 frames
            PointVector points_to_add;
            for (const auto& pd : pt_meas) {
                if (!pd.if_valid) continue;

                Eigen::Quaterniond q;
                spline_.itpQuaternion(pd.time_ns, &q, nullptr, nullptr, nullptr);
                Eigen::Vector3d p = spline_.itpPosition(pd.time_ns, nullptr);
                Eigen::Vector3d pt_w = q * (pd.q_bl * pd.pt_b + pd.t_bl) + p;

                pcl::PointXYZINormal pt;
                pt.x = pt_w.x();
                pt.y = pt_w.y();
                pt.z = pt_w.z();
                points_to_add.push_back(pt);
            }

            if (!points_to_add.empty()) {
                ikd_tree_->Add_Points(points_to_add, true);
                Logger::debug("    Added " + to_string(points_to_add.size()) + " points to map");
            }
        }

        // Save trajectory at the LATEST spline time (where IESKF updated)
        // The spline covers [minTimeNs, maxTimeNs], we interpolate near the end
        // to get the corrected pose from IESKF update
        int64_t query_time = spline_.maxTimeNs() - static_cast<int64_t>(dt_ns);
        if (query_time < spline_.minTimeNs()) {
            query_time = spline_.minTimeNs();
        }

        Eigen::Quaterniond q;
        spline_.itpQuaternion(query_time, &q, nullptr, nullptr, nullptr);
        Eigen::Vector3d p = spline_.itpPosition(query_time, nullptr);

        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose.block<3,3>(0,0) = q.toRotationMatrix();
        pose.block<3,1>(0,3) = p;
        trajectory_.push_back({frame.timestamp, pose});
    }
};

// ========= Main Function =========
int main(int argc, char** argv) {
    cout << "==================================================" << endl;
    cout << "   Gear-Spline LIO Offline Rosbag Test" << endl;
    cout << "==================================================" << endl;

    // Configuration
    string bag_path = "/root/catkin_coco/datasets/degenerate_seq_00.bag";
    string lidar_topic = "/livox/lidar";
    string imu_topic = "/livox/imu";
    string lidar_type = "Livox";

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--bag" && i + 1 < argc) {
            bag_path = argv[++i];
        } else if (arg == "--lidar-topic" && i + 1 < argc) {
            lidar_topic = argv[++i];
        } else if (arg == "--imu-topic" && i + 1 < argc) {
            imu_topic = argv[++i];
        } else if (arg == "--lidar-type" && i + 1 < argc) {
            lidar_type = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            cout << "Usage: " << argv[0] << " [options]" << endl;
            cout << "Options:" << endl;
            cout << "  --bag PATH         Path to rosbag file" << endl;
            cout << "  --lidar-topic NAME LiDAR topic name" << endl;
            cout << "  --imu-topic NAME   IMU topic name" << endl;
            cout << "  --lidar-type TYPE  LiDAR type (Livox/Ouster)" << endl;
            return 0;
        }
    }

    Logger::info("Configuration:");
    Logger::info("  Bag path: " + bag_path);
    Logger::info("  LiDAR topic: " + lidar_topic);
    Logger::info("  IMU topic: " + imu_topic);
    Logger::info("  LiDAR type: " + lidar_type);

    // Load rosbag
    RosbagLoader loader;
    loader.bag_path_ = bag_path;
    loader.lidar_topic_ = lidar_topic;
    loader.imu_topic_ = imu_topic;
    loader.lidar_type_ = lidar_type;

    if (!loader.load()) {
        Logger::error("Failed to load rosbag!");
        return 1;
    }

    // Run IESKF
    OfflineIESKFRunner runner;
    runner.run(loader.imu_data_, loader.lidar_data_);

    // Save trajectory to file
    string traj_path = "trajectory.txt";
    Logger::info("Saving trajectory to " + traj_path);
    ofstream traj_file(traj_path);
    for (const auto& entry : runner.trajectory_) {
        traj_file << fixed << setprecision(6) << entry.first << " ";
        // TUM format: timestamp tx ty tz qx qy qz qw
        Eigen::Quaterniond q(entry.second.block<3,3>(0,0));
        Eigen::Vector3d t = entry.second.block<3,1>(0,3);
        traj_file << t.x() << " " << t.y() << " " << t.z() << " ";
        traj_file << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }
    traj_file.close();

    Logger::info("Done!");
    return 0;
}
