#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>

namespace gear_spline {

/**
 * Gear Mode Enumeration
 * - SPORT: High dynamics, short dt (10ms), 4 control points
 * - NORMAL: Normal dynamics, medium dt (50ms), 8 control points
 * - ECO: Low dynamics, long dt (200ms), 15 control points
 */
enum class GearMode { SPORT = 0, NORMAL = 1, ECO = 2 };

/**
 * Gear Profile Configuration
 */
struct GearProfile {
    GearMode mode;
    double dt;              // Control point time interval (seconds)
    int target_N;           // Target window size (control point count)
    double energy_threshold; // Energy threshold to switch TO this mode (from lower mode)

    GearProfile() : mode(GearMode::NORMAL), dt(0.05), target_N(8), energy_threshold(0.5) {}
    GearProfile(GearMode m, double d, int n, double e)
        : mode(m), dt(d), target_N(n), energy_threshold(e) {}
};

/**
 * GearSystem - Dynamic gear switching based on IMU energy
 *
 * The gear system monitors IMU measurements and switches between
 * SPORT/NORMAL/ECO modes to balance accuracy and efficiency:
 * - High IMU energy (fast motion) -> SPORT mode (fine resolution)
 * - Medium IMU energy -> NORMAL mode
 * - Low IMU energy (slow motion) -> ECO mode (coarse resolution)
 */
class GearSystem {
public:
    GearSystem() {
        // Initialize gear profiles
        profiles_[0] = GearProfile(GearMode::SPORT,  0.01, 4,  2.0);   // 10ms, 4 CPs
        profiles_[1] = GearProfile(GearMode::NORMAL, 0.05, 8,  0.5);   // 50ms, 8 CPs
        profiles_[2] = GearProfile(GearMode::ECO,    0.20, 15, 0.0);   // 200ms, 15 CPs

        current_mode_ = GearMode::NORMAL;
        active_N_ = profiles_[1].target_N;

        // Hysteresis parameters to prevent rapid switching
        upshift_delay_frames_ = 0;
        downshift_delay_frames_ = 0;
        upshift_delay_threshold_ = 2;   // Need 2 consecutive frames to upshift
        downshift_delay_threshold_ = 5; // Need 5 consecutive frames to downshift
    }

    /**
     * Update gear decision based on IMU measurements
     * @param gyros Recent gyroscope measurements
     * @param accels Recent accelerometer measurements
     * @return true if resampling is needed (gear upshift occurred)
     */
    bool updateDecision(const std::vector<Eigen::Vector3d>& gyros,
                        const std::vector<Eigen::Vector3d>& accels) {
        double energy = computeEnergy(gyros, accels);
        last_energy_ = energy;

        GearMode target_mode = determineTargetMode(energy);
        bool needs_resample = false;

        if (target_mode < current_mode_) {
            // Upshift (ECO->NORMAL->SPORT): Safety first, execute immediately
            upshift_delay_frames_++;
            downshift_delay_frames_ = 0;

            if (upshift_delay_frames_ >= upshift_delay_threshold_) {
                current_mode_ = target_mode;
                active_N_ = profiles_[static_cast<int>(target_mode)].target_N;
                needs_resample = true;  // Critical: Must resample on upshift
                upshift_delay_frames_ = 0;
            }
        } else if (target_mode > current_mode_) {
            // Downshift (SPORT->NORMAL->ECO): Delayed transition (allow natural growth)
            downshift_delay_frames_++;
            upshift_delay_frames_ = 0;

            if (downshift_delay_frames_ >= downshift_delay_threshold_) {
                current_mode_ = target_mode;
                // active_N doesn't change immediately, grows naturally
                downshift_delay_frames_ = 0;
            }
        } else {
            // Stay in current mode
            upshift_delay_frames_ = 0;
            downshift_delay_frames_ = 0;
        }

        return needs_resample;
    }

    /**
     * Try to grow the window size towards target_N
     * Called each frame when in downshift mode
     */
    void tryGrowWindow() {
        int target_N = profiles_[static_cast<int>(current_mode_)].target_N;
        if (active_N_ < target_N) {
            active_N_++;
        }
    }

    // Accessors
    GearMode getCurrentMode() const { return current_mode_; }
    double getCurrentDt() const { return profiles_[static_cast<int>(current_mode_)].dt; }
    int64_t getCurrentDtNs() const {
        return static_cast<int64_t>(getCurrentDt() * 1e9);
    }
    int getActiveN() const { return active_N_; }
    int getTargetN() const { return profiles_[static_cast<int>(current_mode_)].target_N; }
    double getLastEnergy() const { return last_energy_; }

    // Set active N directly (for initialization)
    void setActiveN(int n) {
        active_N_ = std::max(4, std::min(n, 15));
    }

    // Force set mode (for testing)
    void setMode(GearMode mode) {
        current_mode_ = mode;
        active_N_ = profiles_[static_cast<int>(mode)].target_N;
    }

    // Get mode string for display
    static std::string modeToString(GearMode mode) {
        switch (mode) {
            case GearMode::SPORT:  return "SPORT";
            case GearMode::NORMAL: return "NORMAL";
            case GearMode::ECO:    return "ECO";
            default: return "UNKNOWN";
        }
    }

    // Get mode color for RViz (RGB)
    static Eigen::Vector3f modeToColor(GearMode mode) {
        switch (mode) {
            case GearMode::SPORT:  return Eigen::Vector3f(1.0f, 0.0f, 0.0f);  // Red
            case GearMode::NORMAL: return Eigen::Vector3f(1.0f, 1.0f, 0.0f);  // Yellow
            case GearMode::ECO:    return Eigen::Vector3f(0.0f, 1.0f, 0.0f);  // Green
            default: return Eigen::Vector3f(0.5f, 0.5f, 0.5f);  // Gray
        }
    }

private:
    GearProfile profiles_[3];
    GearMode current_mode_;
    int active_N_;
    double last_energy_ = 0.0;

    // Hysteresis counters
    int upshift_delay_frames_;
    int downshift_delay_frames_;
    int upshift_delay_threshold_;
    int downshift_delay_threshold_;

    /**
     * Compute motion energy from IMU measurements
     * Uses angular velocity magnitude as primary indicator
     */
    double computeEnergy(const std::vector<Eigen::Vector3d>& gyros,
                         const std::vector<Eigen::Vector3d>& accels) {
        if (gyros.empty()) return 0.0;

        // Primary: Angular velocity energy
        double gyro_energy = 0.0;
        for (const auto& w : gyros) {
            gyro_energy += w.norm();
        }
        gyro_energy /= gyros.size();

        // Secondary: Acceleration variation (optional, for more robust detection)
        double accel_energy = 0.0;
        if (!accels.empty() && accels.size() > 1) {
            Eigen::Vector3d mean_acc = Eigen::Vector3d::Zero();
            for (const auto& a : accels) {
                mean_acc += a;
            }
            mean_acc /= accels.size();

            for (const auto& a : accels) {
                accel_energy += (a - mean_acc).norm();
            }
            accel_energy /= accels.size();
        }

        // Combined energy (primarily gyro-based)
        return gyro_energy + 0.1 * accel_energy;
    }

    /**
     * Determine target mode based on energy level
     */
    GearMode determineTargetMode(double energy) {
        if (energy > profiles_[0].energy_threshold) {
            return GearMode::SPORT;   // High energy -> SPORT
        } else if (energy > profiles_[1].energy_threshold) {
            return GearMode::NORMAL;  // Medium energy -> NORMAL
        } else {
            return GearMode::ECO;     // Low energy -> ECO
        }
    }
};

/**
 * StateResampler - Resample control points when gear shifting
 *
 * When upshifting (e.g., ECO->SPORT), the old control point spacing (dt=200ms)
 * doesn't match the new mode (dt=10ms). We need to resample the trajectory
 * to generate new control points at the correct spacing.
 */
class StateResampler {
public:
    /**
     * Resample control points from old spline to new dt/N configuration
     *
     * @param old_spline The original SplineState
     * @param new_spline Output: resampled SplineState
     * @param current_time_ns Current time (end of trajectory)
     * @param new_dt New control point time interval (seconds)
     * @param new_N New number of control points
     *
     * The resampling process:
     * 1. Calculate new start time based on current_time and new window size
     * 2. For each new control point time, interpolate position/orientation from old spline
     * 3. Convert interpolated quaternions to orientation deltas
     */
    template<typename SplineStateT>
    static void resampleControlPoints(
        const SplineStateT& old_spline,
        SplineStateT& new_spline,
        int64_t current_time_ns,
        double new_dt,
        int new_N)
    {
        int64_t new_dt_ns = static_cast<int64_t>(new_dt * 1e9);

        // Calculate new start time
        int64_t new_start_time = current_time_ns - (new_N - 1) * new_dt_ns;

        // Clamp to old spline's valid range
        int64_t min_time = old_spline.minTimeNs();
        int64_t max_time = old_spline.maxTimeNs();

        // Get initial pose from old spline
        int64_t init_time = std::max(new_start_time, min_time);
        init_time = std::min(init_time, max_time);

        Eigen::Quaterniond q_init;
        old_spline.itpQuaternion(init_time, &q_init);
        Eigen::Vector3d p_init = old_spline.itpPosition(init_time);

        // Initialize new spline
        new_spline.init(new_dt_ns, 0, new_start_time, 0, p_init, q_init);

        // Sample from old spline to create new control points
        Eigen::Quaterniond q_prev = q_init;

        for (int i = 0; i < new_N; i++) {
            int64_t sample_time = new_start_time + i * new_dt_ns;

            // Clamp to valid range
            sample_time = std::max(sample_time, min_time);
            sample_time = std::min(sample_time, max_time);

            // Interpolate position
            Eigen::Vector3d pos = old_spline.itpPosition(sample_time);

            // Interpolate quaternion
            Eigen::Quaterniond q;
            old_spline.itpQuaternion(sample_time, &q);

            // Compute orientation delta from previous quaternion
            Eigen::Quaterniond q_delta = q_prev.inverse() * q;

            // Convert to rotation vector (log map)
            Eigen::Vector3d ort_del = quaternionToRotationVector(q_delta);

            new_spline.addOneStateKnot(pos, ort_del);
            q_prev = q;
        }
    }

    /**
     * In-place resample (modifies the spline directly)
     * This is more efficient when we don't need to keep the old spline
     */
    template<typename SplineStateT>
    static void resampleInPlace(
        SplineStateT& spline,
        int64_t current_time_ns,
        double new_dt,
        int new_N)
    {
        // Store interpolated poses
        std::vector<Eigen::Vector3d> new_positions(new_N);
        std::vector<Eigen::Vector3d> new_ort_deltas(new_N);

        int64_t new_dt_ns = static_cast<int64_t>(new_dt * 1e9);
        int64_t new_start_time = current_time_ns - (new_N - 1) * new_dt_ns;

        int64_t min_time = spline.minTimeNs();
        int64_t max_time = spline.maxTimeNs();

        // Sample from current spline
        Eigen::Quaterniond q_prev;
        spline.itpQuaternion(std::max(new_start_time, min_time), &q_prev);

        for (int i = 0; i < new_N; i++) {
            int64_t sample_time = new_start_time + i * new_dt_ns;
            sample_time = std::max(sample_time, min_time);
            sample_time = std::min(sample_time, max_time);

            new_positions[i] = spline.itpPosition(sample_time);

            Eigen::Quaterniond q;
            spline.itpQuaternion(sample_time, &q);

            Eigen::Quaterniond q_delta = q_prev.inverse() * q;
            new_ort_deltas[i] = quaternionToRotationVector(q_delta);
            q_prev = q;
        }

        // Reinitialize spline with new configuration
        Eigen::Quaterniond q_init;
        spline.itpQuaternion(std::max(new_start_time, min_time), &q_init);

        spline.init(new_dt_ns, 0, new_start_time, 0, new_positions[0], q_init);

        for (int i = 0; i < new_N; i++) {
            spline.addOneStateKnot(new_positions[i], new_ort_deltas[i]);
        }

        // Update active_N
        spline.setActiveN(new_N);
    }

private:
    /**
     * Convert quaternion to rotation vector (log map)
     */
    static Eigen::Vector3d quaternionToRotationVector(const Eigen::Quaterniond& q) {
        Eigen::AngleAxisd aa(q);
        double angle = aa.angle();

        // Handle singularity near identity
        if (std::abs(angle) < 1e-10) {
            return Eigen::Vector3d::Zero();
        }

        // Handle angle wrap-around (prefer smaller rotation)
        if (angle > M_PI) {
            angle -= 2 * M_PI;
        } else if (angle < -M_PI) {
            angle += 2 * M_PI;
        }

        return angle * aa.axis();
    }
};

/**
 * BlendingCache - Pre-compute blending matrices for non-uniform B-splines
 *
 * For non-uniform B-splines, the blending matrix depends on the dt intervals.
 * Computing these on-the-fly in the OMP loop would be expensive.
 * This cache pre-computes them before the frame processing loop.
 */
class BlendingCache {
public:
    struct CacheEntry {
        Eigen::Matrix4d M;           // Blending matrix
        Eigen::Matrix4d M_cumulative; // Cumulative blending matrix (for SO3)
        double dt;                    // Time interval
        bool valid = false;
    };

    BlendingCache() = default;

    /**
     * Pre-compute blending matrices for given dt intervals
     * MUST be called BEFORE the OMP parallel loop
     */
    void precompute(const std::vector<double>& dt_intervals) {
        cache_.resize(dt_intervals.size());
        for (size_t i = 0; i < dt_intervals.size(); i++) {
            cache_[i].dt = dt_intervals[i];
            cache_[i].M = computeBlendingMatrix(dt_intervals[i]);
            cache_[i].M_cumulative = computeCumulativeBlendingMatrix(dt_intervals[i]);
            cache_[i].valid = true;
        }
        is_valid_ = true;
    }

    /**
     * Check if cache is valid
     */
    bool isValid() const { return is_valid_; }

    /**
     * Invalidate cache (call when dt intervals change)
     */
    void invalidate() { is_valid_ = false; }

    /**
     * Get cached blending matrix for given index
     */
    const Eigen::Matrix4d& getMatrix(size_t idx) const {
        assert(idx < cache_.size() && cache_[idx].valid);
        return cache_[idx].M;
    }

    /**
     * Get cached cumulative blending matrix for given index
     */
    const Eigen::Matrix4d& getCumulativeMatrix(size_t idx) const {
        assert(idx < cache_.size() && cache_[idx].valid);
        return cache_[idx].M_cumulative;
    }

    /**
     * Get cache size
     */
    size_t size() const { return cache_.size(); }

private:
    std::vector<CacheEntry> cache_;
    bool is_valid_ = false;

    /**
     * Compute standard uniform B-spline blending matrix
     * For non-uniform case, this would use De Boor algorithm
     */
    static Eigen::Matrix4d computeBlendingMatrix(double dt) {
        // Standard cubic B-spline blending matrix (uniform)
        // M = 1/6 * [1, 4, 1, 0; -3, 0, 3, 0; 3, -6, 3, 0; -1, 3, -3, 1]
        Eigen::Matrix4d M;
        M << 1, 4, 1, 0,
            -3, 0, 3, 0,
             3, -6, 3, 0,
            -1, 3, -3, 1;
        return M / 6.0;
    }

    /**
     * Compute cumulative blending matrix (for SO3 interpolation)
     */
    static Eigen::Matrix4d computeCumulativeBlendingMatrix(double dt) {
        Eigen::Matrix4d M = computeBlendingMatrix(dt);

        // Make cumulative: row_i += sum(row_{i+1}, row_{i+2}, ...)
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                M.row(i) += M.row(j);
            }
        }

        return M;
    }
};

} // namespace gear_spline
