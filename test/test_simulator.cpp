/**
 * Gear-Spline LIO: IMU & LiDAR Simulator
 *
 * Generates synthetic IMU and LiDAR data for testing all modules:
 * - SplineState interpolation
 * - IESKF estimation
 * - GearSystem switching
 * - StateResampler accuracy
 */

#define STANDALONE_TEST

#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>
#include <memory>
#include <chrono>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "gear_spline/SplineState.hpp"
#include "gear_spline/GearSystem.hpp"
#include "gear_spline/Estimator.hpp"

// ============================================================================
// Noise Models
// ============================================================================

class NoiseGenerator {
public:
    NoiseGenerator(uint64_t seed = 42) : gen_(seed) {}

    // Gaussian noise
    double gaussian(double mean, double stddev) {
        std::normal_distribution<double> dist(mean, stddev);
        return dist(gen_);
    }

    Eigen::Vector3d gaussian3d(double stddev) {
        return Eigen::Vector3d(
            gaussian(0, stddev),
            gaussian(0, stddev),
            gaussian(0, stddev)
        );
    }

    // Random walk (for bias drift)
    Eigen::Vector3d randomWalk(Eigen::Vector3d& state, double dt, double sigma) {
        state += gaussian3d(sigma * std::sqrt(dt));
        return state;
    }

private:
    std::mt19937_64 gen_;
};

// ============================================================================
// Trajectory Generator
// ============================================================================

class TrajectoryGenerator {
public:
    enum TrajectoryType {
        STATIC,         // Stationary
        CONSTANT_VEL,   // Constant velocity
        SINUSOIDAL,     // Sinusoidal motion
        CIRCULAR,       // Circular motion
        FIGURE_EIGHT,   // Figure-8 pattern
        AGGRESSIVE      // High-dynamic motion (for gear switching test)
    };

    TrajectoryGenerator(TrajectoryType type = SINUSOIDAL)
        : type_(type), t_(0), dt_(0.001) {}

    // Get pose at time t
    void getPose(double t, Eigen::Vector3d& pos, Eigen::Quaterniond& q,
                 Eigen::Vector3d& vel, Eigen::Vector3d& omega,
                 Eigen::Vector3d& acc) {
        switch (type_) {
            case STATIC:
                getStaticPose(t, pos, q, vel, omega, acc);
                break;
            case CONSTANT_VEL:
                getConstantVelPose(t, pos, q, vel, omega, acc);
                break;
            case SINUSOIDAL:
                getSinusoidalPose(t, pos, q, vel, omega, acc);
                break;
            case CIRCULAR:
                getCircularPose(t, pos, q, vel, omega, acc);
                break;
            case FIGURE_EIGHT:
                getFigureEightPose(t, pos, q, vel, omega, acc);
                break;
            case AGGRESSIVE:
                getAggressivePose(t, pos, q, vel, omega, acc);
                break;
        }
    }

private:
    TrajectoryType type_;
    double t_;
    double dt_;

    void getStaticPose(double t, Eigen::Vector3d& pos, Eigen::Quaterniond& q,
                       Eigen::Vector3d& vel, Eigen::Vector3d& omega, Eigen::Vector3d& acc) {
        pos = Eigen::Vector3d(0, 0, 0);
        q = Eigen::Quaterniond::Identity();
        vel.setZero();
        omega.setZero();
        acc = Eigen::Vector3d(0, 0, 9.81);  // Gravity
    }

    void getConstantVelPose(double t, Eigen::Vector3d& pos, Eigen::Quaterniond& q,
                            Eigen::Vector3d& vel, Eigen::Vector3d& omega, Eigen::Vector3d& acc) {
        double v = 1.0;  // 1 m/s
        pos = Eigen::Vector3d(v * t, 0, 0);
        q = Eigen::Quaterniond::Identity();
        vel = Eigen::Vector3d(v, 0, 0);
        omega.setZero();
        acc = Eigen::Vector3d(0, 0, 9.81);
    }

    void getSinusoidalPose(double t, Eigen::Vector3d& pos, Eigen::Quaterniond& q,
                           Eigen::Vector3d& vel, Eigen::Vector3d& omega, Eigen::Vector3d& acc) {
        double freq = 0.2;  // Hz
        double amp_pos = 2.0;  // m
        double amp_rot = 0.3;  // rad
        double w = 2 * M_PI * freq;

        // Position: sinusoidal in X-Y plane
        pos.x() = amp_pos * std::sin(w * t);
        pos.y() = amp_pos * std::sin(2 * w * t) * 0.5;
        pos.z() = 0;

        // Velocity
        vel.x() = amp_pos * w * std::cos(w * t);
        vel.y() = amp_pos * w * std::cos(2 * w * t);
        vel.z() = 0;

        // Acceleration (in world frame, add gravity)
        acc.x() = -amp_pos * w * w * std::sin(w * t);
        acc.y() = -2 * amp_pos * w * w * std::sin(2 * w * t);
        acc.z() = 9.81;

        // Rotation: sinusoidal yaw
        double yaw = amp_rot * std::sin(w * t);
        q = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());

        // Angular velocity
        omega = Eigen::Vector3d(0, 0, amp_rot * w * std::cos(w * t));
    }

    void getCircularPose(double t, Eigen::Vector3d& pos, Eigen::Quaterniond& q,
                         Eigen::Vector3d& vel, Eigen::Vector3d& omega, Eigen::Vector3d& acc) {
        double radius = 5.0;  // m
        double omega_circle = 0.5;  // rad/s

        pos.x() = radius * std::cos(omega_circle * t);
        pos.y() = radius * std::sin(omega_circle * t);
        pos.z() = 0;

        vel.x() = -radius * omega_circle * std::sin(omega_circle * t);
        vel.y() = radius * omega_circle * std::cos(omega_circle * t);
        vel.z() = 0;

        // Centripetal + gravity
        acc.x() = -radius * omega_circle * omega_circle * std::cos(omega_circle * t);
        acc.y() = -radius * omega_circle * omega_circle * std::sin(omega_circle * t);
        acc.z() = 9.81;

        // Face tangent direction
        double yaw = omega_circle * t + M_PI / 2;
        q = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
        omega = Eigen::Vector3d(0, 0, omega_circle);
    }

    void getFigureEightPose(double t, Eigen::Vector3d& pos, Eigen::Quaterniond& q,
                            Eigen::Vector3d& vel, Eigen::Vector3d& omega, Eigen::Vector3d& acc) {
        double a = 5.0;  // Size
        double w = 0.3;  // Angular frequency

        // Lemniscate of Bernoulli
        double denom = 1 + std::sin(w * t) * std::sin(w * t);
        pos.x() = a * std::cos(w * t) / denom;
        pos.y() = a * std::sin(w * t) * std::cos(w * t) / denom;
        pos.z() = 0;

        // Numerical derivatives
        double dt = 0.001;
        double denom2 = 1 + std::sin(w * (t + dt)) * std::sin(w * (t + dt));
        Eigen::Vector3d pos2;
        pos2.x() = a * std::cos(w * (t + dt)) / denom2;
        pos2.y() = a * std::sin(w * (t + dt)) * std::cos(w * (t + dt)) / denom2;
        pos2.z() = 0;

        vel = (pos2 - pos) / dt;
        acc = Eigen::Vector3d(0, 0, 9.81);  // Simplified

        double yaw = std::atan2(vel.y(), vel.x());
        q = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
        omega = Eigen::Vector3d(0, 0, 0.1);  // Approximate
    }

    void getAggressivePose(double t, Eigen::Vector3d& pos, Eigen::Quaterniond& q,
                           Eigen::Vector3d& vel, Eigen::Vector3d& omega, Eigen::Vector3d& acc) {
        // Alternating between slow and fast motion phases
        double phase = std::fmod(t, 10.0);  // 10s period

        double amp, freq;
        if (phase < 3.0) {
            // Slow phase (ECO mode)
            amp = 0.5;
            freq = 0.1;
        } else if (phase < 6.0) {
            // Medium phase (NORMAL mode)
            amp = 2.0;
            freq = 0.5;
        } else {
            // Fast phase (SPORT mode)
            amp = 5.0;
            freq = 2.0;
        }

        double w = 2 * M_PI * freq;
        pos.x() = amp * std::sin(w * t);
        pos.y() = amp * std::cos(w * t) * 0.5;
        pos.z() = 0;

        vel.x() = amp * w * std::cos(w * t);
        vel.y() = -amp * w * std::sin(w * t) * 0.5;
        vel.z() = 0;

        acc.x() = -amp * w * w * std::sin(w * t);
        acc.y() = -amp * w * w * std::cos(w * t) * 0.5;
        acc.z() = 9.81;

        double yaw = freq * 2 * std::sin(w * t);
        q = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
        omega = Eigen::Vector3d(0, 0, freq * 2 * w * std::cos(w * t));
    }
};

// ============================================================================
// IMU Simulator
// ============================================================================

struct ImuMeasurement {
    int64_t time_ns;
    Eigen::Vector3d accel;  // In body frame
    Eigen::Vector3d gyro;   // In body frame
};

class ImuSimulator {
public:
    struct Config {
        double accel_noise_density;
        double gyro_noise_density;
        double accel_bias_random_walk;
        double gyro_bias_random_walk;
        Eigen::Vector3d accel_bias_init;
        Eigen::Vector3d gyro_bias_init;
        double rate_hz;

        Config() : accel_noise_density(0.01), gyro_noise_density(0.001),
                   accel_bias_random_walk(0.001), gyro_bias_random_walk(0.0001),
                   accel_bias_init(0.1, -0.05, 0.08), gyro_bias_init(0.01, -0.01, 0.005),
                   rate_hz(200.0) {}
    };

    ImuSimulator(const Config& config = Config())
        : config_(config), noise_(42),
          accel_bias_(config.accel_bias_init),
          gyro_bias_(config.gyro_bias_init) {}

    ImuMeasurement generate(double t, TrajectoryGenerator& traj) {
        Eigen::Vector3d pos, vel, omega, acc;
        Eigen::Quaterniond q;
        traj.getPose(t, pos, q, vel, omega, acc);

        // Transform acceleration to body frame
        Eigen::Matrix3d R = q.toRotationMatrix();
        Eigen::Vector3d acc_body = R.transpose() * acc;

        // Add noise and bias
        double dt = 1.0 / config_.rate_hz;
        double accel_noise_std = config_.accel_noise_density * std::sqrt(config_.rate_hz);
        double gyro_noise_std = config_.gyro_noise_density * std::sqrt(config_.rate_hz);

        // Update random walk bias
        noise_.randomWalk(accel_bias_, dt, config_.accel_bias_random_walk);
        noise_.randomWalk(gyro_bias_, dt, config_.gyro_bias_random_walk);

        ImuMeasurement meas;
        meas.time_ns = static_cast<int64_t>(t * 1e9);
        meas.accel = acc_body + accel_bias_ + noise_.gaussian3d(accel_noise_std);
        meas.gyro = omega + gyro_bias_ + noise_.gaussian3d(gyro_noise_std);

        return meas;
    }

    // Get true bias (for evaluation)
    Eigen::Vector3d getAccelBias() const { return accel_bias_; }
    Eigen::Vector3d getGyroBias() const { return gyro_bias_; }

private:
    Config config_;
    NoiseGenerator noise_;
    Eigen::Vector3d accel_bias_;
    Eigen::Vector3d gyro_bias_;
};

// ============================================================================
// LiDAR Point Cloud Simulator
// ============================================================================

struct LidarPoint {
    Eigen::Vector3d position;  // In body frame
    double intensity;
    int64_t time_ns;
};

struct LidarScan {
    int64_t time_ns;
    std::vector<LidarPoint> points;
};

class LidarSimulator {
public:
    struct Config {
        double range_noise_std;
        double range_min;
        double range_max;
        int num_lines;
        int points_per_line;
        double scan_duration_ms;
        double fov_up;
        double fov_down;

        Config() : range_noise_std(0.02), range_min(0.5), range_max(100.0),
                   num_lines(64), points_per_line(1024), scan_duration_ms(100),
                   fov_up(15.0), fov_down(-25.0) {}
    };

    LidarSimulator(const Config& config = Config())
        : config_(config), noise_(123) {}

    // Generate a scan of a simple box environment
    LidarScan generateBoxEnvironment(double t, TrajectoryGenerator& traj) {
        Eigen::Vector3d pos, vel, omega, acc;
        Eigen::Quaterniond q;
        traj.getPose(t, pos, q, vel, omega, acc);

        LidarScan scan;
        scan.time_ns = static_cast<int64_t>(t * 1e9);

        // Box walls
        std::vector<Eigen::Vector4d> planes = {
            {1, 0, 0, 50},   // x = 50
            {-1, 0, 0, 50},  // x = -50
            {0, 1, 0, 50},   // y = 50
            {0, -1, 0, 50},  // y = -50
            {0, 0, 1, 10},   // z = 10 (ceiling)
            {0, 0, -1, 0}    // z = 0 (floor)
        };

        double dt_point = config_.scan_duration_ms * 1e-6 /
                          (config_.num_lines * config_.points_per_line);

        for (int line = 0; line < config_.num_lines; line++) {
            double vert_angle = config_.fov_down +
                (config_.fov_up - config_.fov_down) * line / (config_.num_lines - 1);
            vert_angle *= M_PI / 180.0;

            for (int pt = 0; pt < config_.points_per_line; pt++) {
                double horiz_angle = 2 * M_PI * pt / config_.points_per_line;

                // Ray direction in body frame
                Eigen::Vector3d ray_body(
                    std::cos(vert_angle) * std::cos(horiz_angle),
                    std::cos(vert_angle) * std::sin(horiz_angle),
                    std::sin(vert_angle)
                );

                // Ray in world frame
                Eigen::Vector3d ray_world = q * ray_body;

                // Find intersection with planes
                double min_t = config_.range_max;
                for (const auto& plane : planes) {
                    Eigen::Vector3d n(plane.x(), plane.y(), plane.z());
                    double d = plane.w();
                    double denom = n.dot(ray_world);
                    if (std::abs(denom) > 1e-6) {
                        double t_hit = (d - n.dot(pos)) / denom;
                        if (t_hit > config_.range_min && t_hit < min_t) {
                            min_t = t_hit;
                        }
                    }
                }

                if (min_t < config_.range_max) {
                    // Add noise to range
                    double noisy_range = min_t + noise_.gaussian(0, config_.range_noise_std);
                    noisy_range = std::max(config_.range_min, noisy_range);

                    LidarPoint point;
                    point.position = ray_body * noisy_range;
                    point.intensity = 100.0;
                    point.time_ns = scan.time_ns +
                        static_cast<int64_t>((line * config_.points_per_line + pt) * dt_point * 1e9);
                    scan.points.push_back(point);
                }
            }
        }

        return scan;
    }

private:
    Config config_;
    NoiseGenerator noise_;
};

// ============================================================================
// Test: Full Pipeline Simulation
// ============================================================================

bool test_imu_simulation() {
    std::cout << "\n=== Test: IMU Simulation ===" << std::endl;

    TrajectoryGenerator traj(TrajectoryGenerator::SINUSOIDAL);
    ImuSimulator imu;

    std::vector<ImuMeasurement> measurements;
    for (double t = 0; t < 1.0; t += 0.005) {  // 200 Hz
        measurements.push_back(imu.generate(t, traj));
    }

    std::cout << "  Generated " << measurements.size() << " IMU samples" << std::endl;
    std::cout << "  First sample: accel=" << measurements[0].accel.transpose()
              << ", gyro=" << measurements[0].gyro.transpose() << std::endl;
    std::cout << "  Last sample: accel=" << measurements.back().accel.transpose()
              << ", gyro=" << measurements.back().gyro.transpose() << std::endl;
    std::cout << "  True bias: accel=" << imu.getAccelBias().transpose()
              << ", gyro=" << imu.getGyroBias().transpose() << std::endl;

    // Verify reasonable values
    bool valid = true;
    for (const auto& m : measurements) {
        if (m.accel.norm() < 5.0 || m.accel.norm() > 15.0) {
            valid = false;
            break;
        }
    }

    std::cout << "  Acceleration magnitude check: " << (valid ? "PASSED" : "FAILED") << std::endl;
    return valid;
}

bool test_lidar_simulation() {
    std::cout << "\n=== Test: LiDAR Simulation ===" << std::endl;

    TrajectoryGenerator traj(TrajectoryGenerator::CONSTANT_VEL);
    LidarSimulator::Config config;
    config.num_lines = 16;
    config.points_per_line = 256;
    LidarSimulator lidar(config);

    LidarScan scan = lidar.generateBoxEnvironment(0.0, traj);

    std::cout << "  Generated " << scan.points.size() << " points" << std::endl;
    std::cout << "  Scan time: " << scan.time_ns << " ns" << std::endl;

    if (!scan.points.empty()) {
        std::cout << "  First point: " << scan.points[0].position.transpose() << std::endl;
        std::cout << "  Point time span: "
                  << (scan.points.back().time_ns - scan.points.front().time_ns) / 1e6
                  << " ms" << std::endl;
    }

    bool valid = scan.points.size() > 1000;
    std::cout << "  Point count check: " << (valid ? "PASSED" : "FAILED") << std::endl;
    return valid;
}

bool test_gear_switching_with_simulation() {
    std::cout << "\n=== Test: Gear Switching with Simulated Data ===" << std::endl;

    TrajectoryGenerator traj(TrajectoryGenerator::AGGRESSIVE);
    ImuSimulator imu;
    gear_spline::GearSystem gear;

    std::vector<gear_spline::GearMode> mode_history;
    std::vector<double> energy_history;

    for (double t = 0; t < 12.0; t += 0.05) {  // 20 Hz update
        // Collect IMU measurements
        std::vector<Eigen::Vector3d> gyros;
        std::vector<Eigen::Vector3d> accels;
        for (double t_imu = t; t_imu < t + 0.05; t_imu += 0.005) {
            auto meas = imu.generate(t_imu, traj);
            gyros.push_back(meas.gyro);
            accels.push_back(meas.accel);
        }

        gear.updateDecision(gyros, accels);
        mode_history.push_back(gear.getCurrentMode());
        energy_history.push_back(gear.getLastEnergy());
    }

    // Check mode transitions
    int sport_count = 0, normal_count = 0, eco_count = 0;
    for (auto mode : mode_history) {
        switch (mode) {
            case gear_spline::GearMode::SPORT: sport_count++; break;
            case gear_spline::GearMode::NORMAL: normal_count++; break;
            case gear_spline::GearMode::ECO: eco_count++; break;
        }
    }

    std::cout << "  Mode distribution: SPORT=" << sport_count
              << ", NORMAL=" << normal_count
              << ", ECO=" << eco_count << std::endl;

    // Verify all modes were visited
    bool valid = (sport_count > 0) && (normal_count > 0) && (eco_count > 0);
    std::cout << "  All modes visited: " << (valid ? "PASSED" : "FAILED") << std::endl;
    return valid;
}

bool test_spline_with_simulation() {
    std::cout << "\n=== Test: Spline Estimation with Simulated Data ===" << std::endl;

    TrajectoryGenerator traj(TrajectoryGenerator::SINUSOIDAL);

    // Generate ground truth trajectory points
    std::vector<double> times;
    std::vector<Eigen::Vector3d> gt_positions;
    std::vector<Eigen::Quaterniond> gt_orientations;

    for (double t = 0; t < 1.0; t += 0.05) {  // 20 Hz
        Eigen::Vector3d pos, vel, omega, acc;
        Eigen::Quaterniond q;
        traj.getPose(t, pos, q, vel, omega, acc);
        times.push_back(t);
        gt_positions.push_back(pos);
        gt_orientations.push_back(q);
    }

    // Initialize spline with ground truth
    SplineState spline;
    int64_t dt_ns = 50000000;  // 50 ms
    spline.init(dt_ns, 0, 0, 0, gt_positions[0], gt_orientations[0]);

    // Add control points from ground truth
    Eigen::Quaterniond q_prev = gt_orientations[0];
    for (size_t i = 0; i < gt_positions.size(); i++) {
        Eigen::Quaterniond q_delta = q_prev.inverse() * gt_orientations[i];
        Eigen::AngleAxisd aa(q_delta);
        Eigen::Vector3d ort_del = aa.angle() * aa.axis();
        if (aa.angle() < 1e-10) ort_del.setZero();

        spline.addOneStateKnot(gt_positions[i], ort_del);
        q_prev = gt_orientations[i];
    }

    // Evaluate interpolation error
    double max_pos_error = 0;
    double max_quat_error = 0;

    for (size_t i = 1; i < times.size() - 1; i++) {
        int64_t t_ns = static_cast<int64_t>(times[i] * 1e9);

        Eigen::Vector3d pos_interp = spline.itpPosition(t_ns);
        Eigen::Quaterniond q_interp;
        spline.itpQuaternion(t_ns, &q_interp);

        double pos_err = (pos_interp - gt_positions[i]).norm();
        double quat_err = std::min(
            (q_interp.coeffs() - gt_orientations[i].coeffs()).norm(),
            (q_interp.coeffs() + gt_orientations[i].coeffs()).norm()
        );

        max_pos_error = std::max(max_pos_error, pos_err);
        max_quat_error = std::max(max_quat_error, quat_err);
    }

    std::cout << "  Max position error: " << max_pos_error << " m" << std::endl;
    std::cout << "  Max quaternion error: " << max_quat_error << std::endl;

    // B-spline approximates (not interpolates), so expect some error
    // Position error < 1m, quaternion error < 0.2 (dot product based)
    bool valid = (max_pos_error < 1.0) && (max_quat_error < 0.2);
    std::cout << "  Error bounds check: " << (valid ? "PASSED" : "FAILED") << std::endl;
    return valid;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "Gear-Spline LIO: Simulator Tests" << std::endl;
    std::cout << "============================================" << std::endl;

    int passed = 0;
    int failed = 0;

    if (test_imu_simulation()) passed++; else failed++;
    if (test_lidar_simulation()) passed++; else failed++;
    if (test_gear_switching_with_simulation()) passed++; else failed++;
    if (test_spline_with_simulation()) passed++; else failed++;

    std::cout << "\n============================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "============================================" << std::endl;

    return failed > 0 ? 1 : 0;
}
