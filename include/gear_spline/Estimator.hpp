#pragma once

#include "SplineState.hpp"
#ifndef STANDALONE_TEST
#include "Association.hpp"
#endif
#include "GearSystem.hpp"
#include <chrono>

// Performance timing macros for Milestone 4
#ifdef GEAR_SPLINE_PROFILE
    #define GEAR_TIMER_START(name) auto _timer_##name = std::chrono::high_resolution_clock::now()
    #define GEAR_TIMER_END(name) do { \
        auto _end_##name = std::chrono::high_resolution_clock::now(); \
        auto _dur_##name = std::chrono::duration_cast<std::chrono::microseconds>(_end_##name - _timer_##name).count(); \
        timing_stats_.name##_us += _dur_##name; \
        timing_stats_.name##_count++; \
    } while(0)
#else
    #define GEAR_TIMER_START(name)
    #define GEAR_TIMER_END(name)
#endif

template<int XSIZE>
class Estimator
{
  public:
    static const int CP_SIZE = 24;
    static const int BA_OFFSET = 24;
    static const int BG_OFFSET = 27;  
    int n_iter = 1;

    Estimator() {};

    void setState(int64_t dt_ns, int64_t start_t_ns, const Eigen::Vector3d& t0, const Eigen::Quaterniond& q0, 
        const Eigen::Matrix<double, XSIZE, XSIZE>& Q, const Eigen::Matrix<double, XSIZE, XSIZE>& P)
    {
        spl.init(dt_ns, 0, start_t_ns, 0, t0, q0);
        for (int i = 0; i < 4; i++) {
            spl.addOneStateKnot(t0, Eigen::Vector3d::Zero());
        }        
        cov_sys = Q;
        cov_rcp = P;
        a_mat = Eigen::Matrix<double, XSIZE, XSIZE>::Zero();
        Eigen::Matrix<double, 6, 6> matblock = Eigen::Matrix<double, 6, 6>::Zero();
        matblock.topLeftCorner<3, 3>().setIdentity();
        matblock.bottomRightCorner<3, 3>().setIdentity();

        a_mat.block(0, 6, 6, 6) = matblock;
        a_mat.block(6, 12, 6, 6) = matblock;
        a_mat.block(12, 18, 6, 6) = matblock;
        a_mat.block(18, 0, 3, 3) = - Eigen::Matrix3d::Identity();
        a_mat.block(18, 12, 3, 3) = 2 * Eigen::Matrix3d::Identity();
        a_mat.block(21, 9, 3, 3) = Eigen::Matrix3d::Identity();   
        if constexpr (XSIZE == 30) {
            a_mat.block(BA_OFFSET, BA_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();
            a_mat.block(BG_OFFSET, BG_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();    
        }
    }

    Eigen::Matrix<double, XSIZE, 1> getState()
    {
        Eigen::Matrix<double, CP_SIZE, 1> cps_win = spl.getRCPs();
        Eigen::Matrix<double, XSIZE, 1> state;
        if constexpr (XSIZE == 24) {
            state = cps_win;
        } else {
            state << cps_win, ba, bg;
        }
        return state;
    }

#ifndef STANDALONE_TEST
    void updateIEKFLiDAR(Eigen::aligned_deque<PointData>& pt_meas, KD_TREE<pcl::PointXYZINormal>* ikdtree, const double pt_thresh, const double cov_thresh)
    {
        const Eigen::Matrix<double, XSIZE, XSIZE> cov_prop = cov_rcp;
        Eigen::Matrix<double, XSIZE, 1> rcp_prop = getState();
        bool converged = true;
        int num_tot_eff = 0;
        int t = 0;
        for (int i = 0; i < max_iter; i++) {
            Eigen::Matrix<double, XSIZE, 1> rcpi = getState();
            if (converged) {
                num_tot_eff = 0;
                Association::findCorresp(num_tot_eff, &spl, ikdtree, pt_meas);
            }
            if (num_tot_eff > 0) {
                updateLiDAR(pt_meas, num_tot_eff, rcp_prop, cov_prop, pt_thresh, cov_thresh);
            } else {
                break;
            }
            converged = true;
            Eigen::Matrix<double, XSIZE, 1> state_af = getState();
            if ((state_af - rcpi).norm() > eps) {
                converged = false;
            } else {
                t++;
            }
            if(!t && i == max_iter - 2) {
                converged = true;
            }
            if ((t > n_iter) || (i == max_iter - 1)) {
                cov_rcp = ( Eigen::MatrixXd::Identity(XSIZE, XSIZE) - KH) * cov_prop;
                cov_rcp = 0.5*(cov_rcp + cov_rcp.transpose());
                break;
            }  
        }
    }

    void updateIEKFLiDARInertial(Eigen::aligned_deque<PointData>& pt_meas, KD_TREE<pcl::PointXYZINormal>* ikdtree, const double pt_thresh,
        Eigen::aligned_deque<ImuData>& imu_meas, const Eigen::Vector3d& g, const Eigen::Vector3d& cov_acc, const Eigen::Vector3d& cov_gyro, const double cov_thresh)
    {
        const Eigen::Matrix<double, XSIZE, XSIZE> cov_prop = cov_rcp;
        Eigen::Matrix<double, XSIZE, 1> rcp_prop = getState();
        bool converged = true;
        int num_tot_eff = 0;
        int t = 0;
        for (int i = 0; i < max_iter; i++) {
            Eigen::Matrix<double, XSIZE, 1> rcpi = getState();
            if (converged) {
                num_tot_eff = 0;
                Association::findCorresp(num_tot_eff, &spl, ikdtree, pt_meas);
            }
            if (num_tot_eff > 0 && imu_meas.empty()) {
                updateLiDAR(pt_meas, num_tot_eff, rcp_prop, cov_prop, pt_thresh, cov_thresh);
            } else if (num_tot_eff > 0) {
                updateLiDARInertial(pt_meas, imu_meas, num_tot_eff, rcp_prop, cov_prop, pt_thresh, cov_thresh, g, cov_acc, cov_gyro);
            } else {
                break;
            }
            converged = true;
            Eigen::Matrix<double, XSIZE, 1> state_af = getState();
            if ((state_af - rcpi).norm() > eps) {
                converged = false;
            } else {
                t++;
            }
            if(!t && i == max_iter - 2) {
                converged = true;
            }
            if ((t > n_iter) || (i == max_iter - 1)) {
                cov_rcp = ( Eigen::MatrixXd::Identity(XSIZE, XSIZE) - KH) * cov_prop;
                cov_rcp = 0.5*(cov_rcp + cov_rcp.transpose());
                break;
            }
        }
    }
#endif // STANDALONE_TEST

    void propRCP(int64_t t)
    {
        if (spl.maxTimeNs() >= t) {
            cov_rcp += cov_sys;
        } else {
            while (spl.maxTimeNs() < t) {
                Eigen::Matrix<double, 24, 1> cps_win = spl.getRCPs();
                Eigen::Matrix<double, 6, 1> cp_prop_pos = 2*cps_win.block<6, 1>(12, 0) - cps_win.block<6, 1>(0, 0);
                Eigen::Vector3d delta = cps_win.segment<3>(9);
                spl.addOneStateKnot(cp_prop_pos.head<3>(), delta);
                cov_rcp = a_mat * cov_rcp * a_mat.transpose() + cov_sys;
            }
        }
    }    

    SplineState* getSpline() {
        return &spl;
    }

    // === Milestone 2: Dynamic control point support ===

    // Get current active state size (active_N * 6 + 6 for bias)
    int getActiveStateSize() const {
        return spl.getActiveN() * 6 + 6;
    }

    // Get current active CP size (active_N * 6)
    int getActiveCPSize() const {
        return spl.getActiveN() * 6;
    }

    // Initialize dynamic covariance matrices
    void initDynamicCovariance(int active_N, double pos_cov, double ort_cov, double bias_cov) {
        int cp_size = active_N * 6;
        int total_size = cp_size + 6;  // +6 for bias

        cov_rcp_dyn = Eigen::MatrixXd::Zero(total_size, total_size);
        cov_sys_dyn = Eigen::MatrixXd::Zero(total_size, total_size);

        // Initialize control point covariance
        for (int i = 0; i < active_N; i++) {
            cov_rcp_dyn.block(i*6, i*6, 3, 3) = pos_cov * Eigen::Matrix3d::Identity();
            cov_rcp_dyn.block(i*6+3, i*6+3, 3, 3) = ort_cov * Eigen::Matrix3d::Identity();
        }

        // Initialize bias covariance
        cov_rcp_dyn.block(cp_size, cp_size, 3, 3) = bias_cov * Eigen::Matrix3d::Identity();
        cov_rcp_dyn.block(cp_size+3, cp_size+3, 3, 3) = bias_cov * Eigen::Matrix3d::Identity();

        // Process noise (smaller for existing CPs, larger for new CPs)
        initDynamicProcessNoise(active_N, pos_cov * 0.01, ort_cov * 0.01);

        use_dynamic_cov_ = true;
        active_N_ = active_N;
    }

    // Initialize dynamic process noise matrix
    void initDynamicProcessNoise(int active_N, double pos_noise, double ort_noise) {
        int cp_size = active_N * 6;
        int total_size = cp_size + 6;

        cov_sys_dyn = Eigen::MatrixXd::Zero(total_size, total_size);

        // Old control points have smaller process noise
        for (int i = 0; i < active_N - 1; i++) {
            cov_sys_dyn.block(i*6, i*6, 3, 3) = pos_noise * 0.1 * Eigen::Matrix3d::Identity();
            cov_sys_dyn.block(i*6+3, i*6+3, 3, 3) = ort_noise * 0.1 * Eigen::Matrix3d::Identity();
        }

        // New control point has larger process noise
        int last = active_N - 1;
        cov_sys_dyn.block(last*6, last*6, 3, 3) = pos_noise * Eigen::Matrix3d::Identity();
        cov_sys_dyn.block(last*6+3, last*6+3, 3, 3) = ort_noise * Eigen::Matrix3d::Identity();
    }

    // Shrink covariance when sliding window drops oldest control point
    void shrinkCovariance() {
        if (!use_dynamic_cov_) return;

        int old_size = cov_rcp_dyn.rows();
        int new_size = old_size - 6;

        if (new_size < 30) return;  // Minimum 4 CPs + bias

        // Keep bottom-right corner (drop oldest control point's covariance)
        Eigen::MatrixXd new_cov(new_size, new_size);
        new_cov = cov_rcp_dyn.bottomRightCorner(new_size, new_size);
        cov_rcp_dyn = new_cov;

        // Update process noise matrix
        Eigen::MatrixXd new_sys(new_size, new_size);
        new_sys = cov_sys_dyn.bottomRightCorner(new_size, new_size);
        cov_sys_dyn = new_sys;

        active_N_--;
    }

    // Expand covariance when window grows to add new control point
    // Uses prediction formula: P_new = A_pred * P_tail * A_pred^T + Q
    void expandCovariance() {
        if (!use_dynamic_cov_) return;

        int old_size = cov_rcp_dyn.rows();
        int new_size = old_size + 6;

        // Check MAX_N limit
        if ((new_size - 6) / 6 > SplineState::getMaxN()) return;

        Eigen::MatrixXd new_cov = Eigen::MatrixXd::Zero(new_size, new_size);

        // Keep original covariance
        new_cov.topLeftCorner(old_size, old_size) = cov_rcp_dyn;

        // Initialize new control point covariance using prediction formula
        // new_pos = 2 * last - second_last
        // A_pred = [2*I, -I] for [last, second_last]
        int cp_size_old = old_size - 6;  // Exclude bias

        if (cp_size_old >= 12) {  // At least 2 control points
            Eigen::Matrix<double, 6, 6> P_tail = cov_rcp_dyn.block<6, 6>(cp_size_old - 6, cp_size_old - 6);
            Eigen::Matrix<double, 6, 6> P_second = cov_rcp_dyn.block<6, 6>(cp_size_old - 12, cp_size_old - 12);
            Eigen::Matrix<double, 6, 6> P_cross = cov_rcp_dyn.block<6, 6>(cp_size_old - 12, cp_size_old - 6);

            // P_new = 4*P_tail + P_second - 4*P_cross + Q
            Eigen::Matrix<double, 6, 6> P_new = 4.0 * P_tail + P_second - 4.0 * P_cross;
            P_new += getProcessNoiseBlock();

            new_cov.block<6, 6>(old_size - 6, old_size - 6) = P_new;  // New CP before bias

            // Cross-covariance: P(new, old) = 2*P(tail, :) - P(second, :)
            Eigen::MatrixXd cross_row = 2.0 * cov_rcp_dyn.block(cp_size_old - 6, 0, 6, old_size)
                                       - cov_rcp_dyn.block(cp_size_old - 12, 0, 6, old_size);
            new_cov.block(old_size - 6, 0, 6, old_size) = cross_row;
            new_cov.block(0, old_size - 6, old_size, 6) = cross_row.transpose();
        } else {
            // Not enough CPs, use default large covariance
            new_cov.block<6, 6>(old_size - 6, old_size - 6) = 0.1 * Eigen::Matrix<double, 6, 6>::Identity();
        }

        // Move bias covariance
        new_cov.block<6, 6>(new_size - 6, new_size - 6) = cov_rcp_dyn.block<6, 6>(old_size - 6, old_size - 6);

        cov_rcp_dyn = new_cov;

        // Update process noise matrix
        Eigen::MatrixXd new_sys = Eigen::MatrixXd::Zero(new_size, new_size);
        new_sys.topLeftCorner(old_size, old_size) = cov_sys_dyn;
        new_sys.block<6, 6>(old_size - 6, old_size - 6) = getProcessNoiseBlock();
        new_sys.block<6, 6>(new_size - 6, new_size - 6) = cov_sys_dyn.block<6, 6>(old_size - 6, old_size - 6);
        cov_sys_dyn = new_sys;

        active_N_++;
    }

    // Get process noise block for new control point
    Eigen::Matrix<double, 6, 6> getProcessNoiseBlock() const {
        Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
        Q.topLeftCorner<3, 3>() = 0.01 * Eigen::Matrix3d::Identity();     // Position noise
        Q.bottomRightCorner<3, 3>() = 0.001 * Eigen::Matrix3d::Identity(); // Orientation noise
        return Q;
    }

    // Rebuild A matrix for dynamic active_N
    void rebuildAMatrix(int active_N) {
        int cp_size = active_N * 6;
        int total_size = cp_size + 6;  // +6 for bias

        a_mat_dyn = Eigen::MatrixXd::Zero(total_size, total_size);
        Eigen::Matrix<double, 6, 6> I6 = Eigen::Matrix<double, 6, 6>::Identity();

        // Window sliding: CP[i] <- CP[i+1]
        for (int i = 0; i < active_N - 1; i++) {
            a_mat_dyn.block(i * 6, (i + 1) * 6, 6, 6) = I6;
        }

        // New control point prediction: new_pos = 2*last - first
        int last = active_N - 1;
        a_mat_dyn.block(last * 6, 0, 3, 3) = -Eigen::Matrix3d::Identity();
        a_mat_dyn.block(last * 6, (active_N - 2) * 6, 3, 3) = 2.0 * Eigen::Matrix3d::Identity();

        // New control point orientation delta: use second-to-last delta
        a_mat_dyn.block(last * 6 + 3, (active_N - 2) * 6 + 3, 3, 3) = Eigen::Matrix3d::Identity();

        // Bias remains unchanged
        a_mat_dyn.block(cp_size, cp_size, 6, 6) = Eigen::Matrix<double, 6, 6>::Identity();
    }

    // Propagate RCP with Gear support (dynamic active_N)
    void propRCPWithGear(int64_t t, int target_N) {
        if (spl.maxTimeNs() >= t) {
            if (use_dynamic_cov_) {
                cov_rcp_dyn += cov_sys_dyn;
            } else {
                cov_rcp += cov_sys;
            }
            return;
        }

        int active_N = spl.getActiveN();

        while (spl.maxTimeNs() < t) {
            if (active_N >= target_N) {
                // Already at target N, perform sliding window
                // Note: SplineState automatically manages sliding via addOneStateKnot
                shrinkCovariance();
            } else {
                // Haven't reached target N, grow window
                spl.growWindow();
                expandCovariance();
                active_N++;
            }

            // Predict new control point
            Eigen::VectorXd cps = spl.getRCPsDynamic();
            int n = cps.rows() / 6;

            Eigen::Vector3d new_pos;
            Eigen::Vector3d new_delta;

            if (n >= 2) {
                new_pos = 2.0 * cps.segment<3>((n - 1) * 6) - cps.segment<3>((n - 2) * 6);
                new_delta = cps.segment<3>((n - 1) * 6 + 3);
            } else {
                new_pos = cps.segment<3>((n - 1) * 6);
                new_delta = Eigen::Vector3d::Zero();
            }

            spl.addOneStateKnot(new_pos, new_delta);

            // Rebuild A matrix and propagate covariance
            if (use_dynamic_cov_) {
                rebuildAMatrix(active_N);
                cov_rcp_dyn = a_mat_dyn * cov_rcp_dyn * a_mat_dyn.transpose() + cov_sys_dyn;
            } else {
                cov_rcp = a_mat * cov_rcp * a_mat.transpose() + cov_sys;
            }
        }

        active_N_ = active_N;
    }

    // Check if using dynamic covariance mode
    bool isUsingDynamicCovariance() const { return use_dynamic_cov_; }

    // Get dynamic covariance matrix (for debugging)
    const Eigen::MatrixXd& getDynamicCovariance() const { return cov_rcp_dyn; }

    // === Milestone 4: Performance Optimization ===

    /**
     * Pre-compute blending cache BEFORE OMP parallel loops
     * This is CRITICAL for non-uniform B-spline performance
     *
     * Call pattern:
     *   1. Collect measurements
     *   2. gear_system.updateDecision()
     *   3. estimator.precomputeBlendingCache()  // <-- Must be here!
     *   4. estimator.updateIEKFLiDARInertial()  // Contains OMP parallel loops
     */
    void precomputeBlendingCache() {
        GEAR_TIMER_START(precompute);

        // Get dt history from spline state
        const std::vector<double>& dt_history = spl.getDtHistory();

        if (!dt_history.empty() && !spl.isUniform()) {
            // Non-uniform mode: precompute blending matrices
            blending_cache_.precompute(dt_history);
        } else {
            // Uniform mode: use standard matrices (already compiled-in)
            // Still mark cache as valid to avoid checks in hot path
            std::vector<double> uniform_dt(spl.getActiveN(),
                static_cast<double>(spl.getKnotTimeIntervalNs()) / 1e9);
            blending_cache_.precompute(uniform_dt);
        }

        GEAR_TIMER_END(precompute);
    }

    /**
     * Check if blending cache is valid (for debugging)
     */
    bool isBlendingCacheValid() const {
        return blending_cache_.isValid();
    }

    /**
     * Invalidate blending cache (call when dt intervals change, e.g., gear shift)
     */
    void invalidateBlendingCache() {
        blending_cache_.invalidate();
    }

    /**
     * Get blending cache reference (for SplineState to use in interpolation)
     */
    const gear_spline::BlendingCache& getBlendingCache() const {
        return blending_cache_;
    }

    // === Performance Statistics ===

    struct TimingStats {
        // Timing accumulators (microseconds)
        uint64_t precompute_us = 0;
        uint64_t lidar_jacobian_us = 0;
        uint64_t imu_jacobian_us = 0;
        uint64_t kalman_update_us = 0;

        // Call counts
        uint64_t precompute_count = 0;
        uint64_t lidar_jacobian_count = 0;
        uint64_t imu_jacobian_count = 0;
        uint64_t kalman_update_count = 0;

        void reset() {
            precompute_us = lidar_jacobian_us = imu_jacobian_us = kalman_update_us = 0;
            precompute_count = lidar_jacobian_count = imu_jacobian_count = kalman_update_count = 0;
        }

        // Get average timing in microseconds
        double avgPrecomputeUs() const {
            return precompute_count > 0 ? (double)precompute_us / precompute_count : 0;
        }
        double avgLidarJacobianUs() const {
            return lidar_jacobian_count > 0 ? (double)lidar_jacobian_us / lidar_jacobian_count : 0;
        }
        double avgImuJacobianUs() const {
            return imu_jacobian_count > 0 ? (double)imu_jacobian_us / imu_jacobian_count : 0;
        }
        double avgKalmanUpdateUs() const {
            return kalman_update_count > 0 ? (double)kalman_update_us / kalman_update_count : 0;
        }
    };

    const TimingStats& getTimingStats() const { return timing_stats_; }
    void resetTimingStats() { timing_stats_.reset(); }

  private:
    // Milestone 4: BlendingCache for non-uniform B-spline optimization
    gear_spline::BlendingCache blending_cache_;
    TimingStats timing_stats_;
    // Dynamic covariance members for Milestone 2
    Eigen::MatrixXd cov_rcp_dyn;
    Eigen::MatrixXd cov_sys_dyn;
    Eigen::MatrixXd a_mat_dyn;
    bool use_dynamic_cov_ = false;
    int active_N_ = 4;

  private:
    SplineState spl;
    Eigen::Matrix<double, XSIZE, XSIZE> cov_rcp;  
    Eigen::Matrix<double, XSIZE, XSIZE> cov_sys; 
    Eigen::Matrix<double, XSIZE, XSIZE> a_mat;   
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba = Eigen::Vector3d::Zero();     
    Eigen::Matrix<double, XSIZE, XSIZE> KH;
    int max_iter = 5;
    double eps = 0.1;

#ifndef STANDALONE_TEST
    void prepIMU(ImuData& imu_data, const Eigen::Vector3d& g)
    {
        Eigen::Quaterniond q_itp;
        Eigen::Vector3d rot_vel;
        Jacobian43 J_ortdel;
        Jacobian J_line_acc;
        Jacobian33 J_gyro;
        spl.itpQuaternion(imu_data.time_ns, &q_itp, &rot_vel, &J_ortdel, &J_gyro);
        Eigen::Vector3d a_w_no_g = spl.itpPosition<2>(imu_data.time_ns, &J_line_acc);
        Eigen::Vector3d a_w = a_w_no_g + g;
        Eigen::Matrix3d RT = q_itp.inverse().toRotationMatrix();   
        Eigen::Matrix<double, 3, 4> drot;
        Quater::drot(a_w, q_itp, drot);
        Eigen::Matrix<double, 6, XSIZE> Hi = Eigen::Matrix<double, 6, XSIZE>::Zero();
        int recur_st_id = spl.numKnots() - 4;
        for (int i = 0; i < (int) J_line_acc.d_val_d_knot.size(); i++) {
            int j = J_line_acc.start_idx + i - recur_st_id;
            if (j >= 0) {
                Hi.block(0, j*6, 3, 3) = RT * J_line_acc.d_val_d_knot[i];
                Hi.block(0, j*6 + 3, 3, 3) = drot * J_ortdel.d_val_d_knot[i];
                Hi.block(3, j*6 + 3, 3, 3) = J_gyro.d_val_d_knot[i];                
            }
        }       
        Hi.block(0, BA_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();
        Hi.block(3, BG_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();            
        imu_data.imu_itp.head<3>() = RT * a_w + ba;
        imu_data.imu_itp.tail<3>() = rot_vel + bg;
        imu_data.H = Hi.template leftCols<24>();
    }    

    void prepLiDAR(PointData& pt_data) const    
    {
        if (pt_data.if_valid) { 
            Eigen::Matrix<double, 1, XSIZE> Hi = Eigen::Matrix<double, 1, XSIZE>::Zero();
            Eigen::Quaterniond q_itp;
            Jacobian43 J_ortdel;
            Jacobian J_pos;
            spl.itpQuaternion(pt_data.time_ns, &q_itp, nullptr, &J_ortdel);
            Eigen::Vector3d p_itp = spl.itpPosition(pt_data.time_ns, &J_pos);
            Eigen::Matrix3d R_IL = pt_data.q_bl.toRotationMatrix();
            Eigen::Vector3d pt_w = q_itp * (R_IL * pt_data.pt_b + pt_data.t_bl) + p_itp;
            pt_data.zp = pt_data.normvec.dot(pt_w) + pt_data.dist;

            Eigen::Matrix<double, 3, 4> drot;
            Quater::drotInv((R_IL * pt_data.pt_b + pt_data.t_bl), q_itp, drot);
            Eigen::Matrix<double, 1, 4> tmp = pt_data.normvec.transpose() * drot;
            int RCP_st_id = spl.numKnots() - 4;
            for (int i = 0; i < (int) J_pos.d_val_d_knot.size(); i++) {
                int j = (int) J_pos.start_idx + i - RCP_st_id;
                if (j >= 0) {
                    Hi.block(0, j*6, 1, 3) = pt_data.normvec.transpose() * J_pos.d_val_d_knot[i];
                    Hi.block(0, j*6 + 3, 1, 3) = tmp * J_ortdel.d_val_d_knot[i];
                }
            }  
            pt_data.H = Hi.template leftCols<24>();
        }
    } 

    void updateState(const Eigen::Matrix<double, XSIZE, 1>& xupd)
    {
        Eigen::Matrix<double, CP_SIZE, 1> cp_win = xupd.segment(0, CP_SIZE);
        spl.updateRCPs(cp_win);
        if constexpr (XSIZE == 30) {
            ba = xupd.segment(BA_OFFSET, 3);
            bg = xupd.segment(BG_OFFSET, 3);   
        }
    }        

    bool updateLiDAR(Eigen::aligned_deque<PointData>& pt_meas, int num_valid, const Eigen::Matrix<double, XSIZE, 1>& x_prop, 
        const Eigen::Matrix<double, XSIZE, XSIZE>& P_prop, const double pt_thresh, const double cov_thresh)
    {
        Eigen::Matrix<double, Eigen::Dynamic, XSIZE> H(num_valid, XSIZE);
        Eigen::Matrix<double, Eigen::Dynamic, 1> innv(num_valid, 1);
        Eigen::Matrix<double, Eigen::Dynamic, 1> mat_cov_inv(num_valid, 1);
        H.setZero();    
        innv.setZero();
        mat_cov_inv.setConstant(1/0.01);
        size_t num_pt = pt_meas.size();    
        #pragma omp parallel for num_threads(NUM_OF_THREAD) schedule(dynamic)
        for (size_t i = 0; i < num_pt; i++) {
            PointData& pt_data = pt_meas[i];
            prepLiDAR(pt_data);
        }        
        int idx_offset = 0;
        for(size_t i = 0; i < num_pt; i++) {
            const PointData& pt_data = pt_meas[i];
            if (pt_data.if_valid) {
                Eigen::Matrix<double, 1, XSIZE> Hi = Eigen::Matrix<double, 1, XSIZE>::Zero();
                Hi.template leftCols<24>() = pt_data.H;
                double lid_cov = Hi*cov_rcp*Hi.transpose() + pt_data.var_pt;
                if (abs(pt_data.zp) < pt_thresh || lid_cov < pt_data.var_pt*cov_thresh) {
                    innv(idx_offset) = - pt_data.zp;
                    H.row(idx_offset) = Hi;
                    
                } 
                mat_cov_inv(idx_offset) = 1/pt_data.var_pt;
                idx_offset++;
            }
        }        
        update(innv, mat_cov_inv, H, x_prop, P_prop);
        return true;
    }           

    void updateLiDARInertial(Eigen::aligned_deque<PointData>& pt_meas, Eigen::aligned_deque<ImuData>& imu_meas, int num_valid, const Eigen::Matrix<double, XSIZE, 1>& x_prop, 
        const Eigen::Matrix<double, XSIZE, XSIZE>& P_prop, const double pt_thresh, const double cov_thresh, const Eigen::Vector3d& g, const Eigen::Vector3d& cov_acc, const Eigen::Vector3d& cov_gyro)
    {
        Eigen::Matrix<double, 6, 1> cov_imu_inv;
        cov_imu_inv << 1/cov_acc[0], 1/cov_acc[1], 1/cov_acc[2], 1/cov_gyro[0], 1/cov_gyro[1], 1/cov_gyro[2];
        #pragma omp parallel for num_threads(NUM_OF_THREAD) schedule(dynamic)
        for (size_t i = 0; i < pt_meas.size(); i++) {
            PointData& pt_data = pt_meas[i];
            prepLiDAR(pt_data); 
        }
        #pragma omp parallel for num_threads(NUM_OF_THREAD) 
        for (size_t i = 0; i < imu_meas.size(); i++) {
            prepIMU(imu_meas[i], g);
        }                
        int dim_meas = 6*imu_meas.size() + num_valid;
        Eigen::Matrix<double, Eigen::Dynamic, XSIZE> H(dim_meas, XSIZE);
        Eigen::Matrix<double, Eigen::Dynamic, 1> innv(dim_meas, 1);
        Eigen::Matrix<double, Eigen::Dynamic, 1> mat_cov_inv(dim_meas, 1);
        H.setZero();    
        innv.setZero();
        mat_cov_inv.setZero();
        int idx_offset = 0;
        size_t id_imu = 0;
        size_t id_pt = 0;
        for (size_t j = 0; j < imu_meas.size() + pt_meas.size(); j++) {
            if ((id_pt < pt_meas.size() && id_imu < imu_meas.size() && pt_meas[id_pt].time_ns < imu_meas[id_imu].time_ns) ||
                (id_pt < pt_meas.size() && id_imu >= imu_meas.size())) {
                    PointData& pt_data = pt_meas[id_pt];
                    if (pt_data.if_valid) {
                        Eigen::Matrix<double, 24, 24> cov = cov_rcp.template topLeftCorner<24, 24>();
                        double lid_cov = pt_data.H*cov*pt_data.H.transpose() + pt_data.var_pt;
                        if (abs(pt_data.zp) < pt_thresh || lid_cov < pt_data.var_pt*cov_thresh) {
                            innv(idx_offset) = - pt_data.zp;
                            H.block(idx_offset, 0, 1, 24) = pt_data.H;
                        } 
                        mat_cov_inv(idx_offset) = 1/pt_data.var_pt;
                        idx_offset++;
                    }
                    id_pt++;
            } else if ((id_pt < pt_meas.size() && id_imu < imu_meas.size() && pt_meas[id_pt].time_ns >= imu_meas[id_imu].time_ns) ||
                        (id_pt >= pt_meas.size() && id_imu < imu_meas.size())) {
                    const ImuData& imu_data = imu_meas[id_imu];
                    Eigen::Matrix<double, 6, 1> imu_itp = imu_data.imu_itp;
                    Eigen::Matrix<double, 6, XSIZE> Hi = Eigen::Matrix<double, 6, XSIZE>::Zero();
                    Hi.template leftCols<24>() = imu_data.H;
                    Hi.block(0, BA_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();
                    Hi.block(3, BG_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();   

                    Eigen::Matrix<double, 6, 1> imu;
                    imu.head<3>() = imu_data.accel;
                    imu.tail<3>() = imu_data.gyro;
                    for (int i = 0; i < 3; i++) {
                        if (abs(imu(i) - imu_itp(i)) > 10.0) {
                            imu(i) = 0;
                            imu_itp(i) = 0;
                            Hi.row(i).setZero();
                        } 
                        if (abs(imu(i+3) - imu_itp(i+3)) > 5.0) {
                            imu(i+3) = 0;
                            imu_itp(i+3) = 0;
                            Hi.row(i+3).setZero();
                        }                     
                    }
                    innv.segment<6>(idx_offset) = imu - imu_itp;
                    H.block(idx_offset, 0, 6, XSIZE) = Hi;
                    mat_cov_inv.segment<6>(idx_offset) = cov_imu_inv;  
                    idx_offset += 6;
                    id_imu++;
            }
        }
        update(innv, mat_cov_inv, H, x_prop, P_prop);
    }
#endif // STANDALONE_TEST

    template <int RSIZE>
    void update(const Eigen::Matrix<double, RSIZE, 1>& innov, const Eigen::Matrix<double, RSIZE, 1>& R_inv, const Eigen::Matrix<double, RSIZE, XSIZE>& H,
        const Eigen::Matrix<double, XSIZE, 1>& x_prop, const Eigen::Matrix<double, XSIZE, XSIZE>& cov_prop)
    {
        int num_pts = innov.rows();
        Eigen::Matrix<double, XSIZE, 1> RCPs_post;
        Eigen::MatrixXd I_X = Eigen::MatrixXd::Identity(XSIZE, XSIZE);

        // Regularization constant for numerical stability
        const double REG_EPSILON = 1e-6;

        if (num_pts > XSIZE) {
            // Add regularization to ensure positive definiteness
            Eigen::Matrix<double, XSIZE, XSIZE> cov_reg = cov_prop;
            cov_reg.diagonal().array() += REG_EPSILON;

            Eigen::LLT<Eigen::Matrix<double, XSIZE, XSIZE>> llt_cov(cov_reg);
            if (llt_cov.info() != Eigen::Success) {
                // Fallback: skip update if decomposition fails
                return;
            }
            Eigen::Matrix<double, XSIZE, XSIZE> cov_rcp_inv = llt_cov.solve(I_X);

            Eigen::Matrix<double, XSIZE, RSIZE> HT_R_inv;
            HT_R_inv.noalias() = (H.transpose().array().rowwise() * R_inv.transpose().array()).matrix();
            Eigen::Matrix<double, XSIZE, XSIZE> HT_R_inv_H;
            HT_R_inv_H.noalias() = HT_R_inv * H;

            Eigen::Matrix<double, XSIZE, XSIZE> S = HT_R_inv_H;
            S.noalias() += cov_rcp_inv;
            S.diagonal().array() += REG_EPSILON;  // Regularize S as well

            Eigen::LLT<Eigen::Matrix<double, XSIZE, XSIZE>> llt_S(S);
            if (llt_S.info() != Eigen::Success) {
                return;
            }
            Eigen::Matrix<double, XSIZE, XSIZE> S_inv = llt_S.solve(I_X);

            Eigen::Matrix<double, XSIZE, RSIZE> K;
            K.noalias() = S_inv * HT_R_inv;

            KH.noalias() = S_inv * HT_R_inv_H;
            Eigen::Matrix<double, XSIZE, 1> delta_cur = (getState() - x_prop);
            Eigen::Matrix<double, XSIZE, 1> deltax = KH * delta_cur + K * innov - delta_cur;
            RCPs_post.noalias() = getState() + deltax;
        } else {
            Eigen::Matrix<double, RSIZE, RSIZE> R = R_inv.cwiseInverse().asDiagonal();
            Eigen::Matrix<double, RSIZE, RSIZE> S;
            S.noalias() = H * cov_prop * H.transpose() + R;
            S.diagonal().array() += REG_EPSILON;  // Regularize for stability

            Eigen::Matrix<double, XSIZE, RSIZE> K;
            K.noalias() = cov_prop * H.transpose() * S.inverse();
            KH.noalias() = K * H;
            Eigen::Matrix<double, XSIZE, 1> delta_cur = (getState() - x_prop);
            Eigen::Matrix<double, XSIZE, 1> deltax = KH * delta_cur + K * innov - delta_cur;
            RCPs_post.noalias() = getState() + deltax;
        }

        // Check for NaN before updating state
        if (RCPs_post.allFinite()) {
            updateState(RCPs_post);
        }
    }    
};
