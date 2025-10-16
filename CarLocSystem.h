#ifndef CAR_LOC_SYSTEM_3D_H
#define CAR_LOC_SYSTEM_3D_H

#include <string>
#include <vector>
#include <memory>
#include <open3d/Open3D.h>

class __declspec(dllexport) CarLocSystem3D {

public:
	CarLocSystem3D() = default;

	/// @brief initialize the system
	/// @param data_dir given data directory
	/// @param logo robot logo
	/// @return initialization success or not
	bool init(std::string data_dir, std::string robot_name);

	/// @brief call point cloud registration to calculate car pose
	/// @param online_pcd_ptr online point cloud ptr
	/// @return calculation success or not
	bool calculateCarPose(std::shared_ptr<open3d::geometry::PointCloud> online_pcd_ptr);
	

	/// @brief convert Eigen::Matrix4d to XYZWPR.
	/// @return pose in vector order: X Y Z in mm, W P R in degree.
	static Eigen::Vector6d toXYZWPR(const Eigen::Matrix4d& pose);
	
	/// @brief 6-dof pose optimization
	/// @param vpose input poses
	/// @return global optimized pose observed by all cameras
	static Eigen::Matrix4d jointlyOptimize(const std::vector<Eigen::Matrix4d>& vpose);

	/// @brief get robot modify pose observed by current camera
	Eigen::Matrix4d getRobotModifyPose();
    /// @brief get car inspect pose observed by current camera
	Eigen::Matrix4d getCarInspectPose();

	/// @brief if the online point cloud is similar to the template point cloud
	bool similarityJudgment(const std::shared_ptr<open3d::geometry::PointCloud>& online_pcd_ptr);

private:

	bool calculateCarPoseImpl(std::shared_ptr<open3d::geometry::PointCloud>& source_pcd_ptr,
		std::shared_ptr<open3d::geometry::PointCloud>& target_pcd_ptr,
		Eigen::Matrix4d& pose_target_source, bool vis_flag = false);

	std::string _robot_name;
	// init data
	std::shared_ptr<open3d::geometry::PointCloud> _cad_pcd_ptr;		// 零偏CAD点云
	std::shared_ptr<open3d::geometry::PointCloud> _cam0_pcd_ptr;	// 零偏实拍点云
	//cam0相对cad
	Eigen::Matrix4d _pose_cad_cam0 = Eigen::Matrix4d::Identity();	// 零偏实拍点云与CAD点云配准结果（预配准或本地读取）
	//base2相对cam0
	Eigen::Matrix4d _pose_bc = Eigen::Matrix4d::Identity();			// 眼在手外


	// 在线配准结果
	//cam1相对cam0
	Eigen::Matrix4d _pose_cam0_cam1 = Eigen::Matrix4d::Identity();	// 当前实拍点云与零偏点云配准结果
	bool _reg_flag = false;
};

#endif // !CAR_LOC_SYSTEM_3D_H
