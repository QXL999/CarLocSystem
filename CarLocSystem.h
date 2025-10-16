#ifndef CAR_LOC_SYSTEM_3D_H
#define CAR_LOC_SYSTEM_3D_H
//思灵
#include <string>
#include <vector>
#include <memory>
#include <open3d/Open3D.h>

/*
	-左相机和右相机的定义：
	--工控机左侧相机为左相机，右侧相机为右相机
	
	-车身正反定义：
	--车身可正/反进入工位，这里使用左/右侧来区分车身正反
	--车身左侧靠近相机则记为左侧，右侧靠近相机则记为右侧
*/


enum class SIDE_ENUM {
	Unknown = 0,
	Left = 1,					// 车身左侧 or 工控机左侧的相机
	Right = 2					// 车身右侧 or 工控机右侧的相机
};

enum class PART_ENUM {
	Unknown = 0,
	FrontLongitudinalBeam = 1,	// 前纵梁
    RearWheelHouse = 2			// 后轮罩
};

/// @brief Template point cloud info (used unit is mm)
struct PointCloudInfo {

	// original template point cloud ptr 模板点云
	std::shared_ptr<open3d::geometry::PointCloud> pcd_ptr = std::make_shared<open3d::geometry::PointCloud>();
	// bounding box of the template point cloud
	Eigen::Vector3d min_bound = Eigen::Vector3d::Zero();
	Eigen::Vector3d max_bound = Eigen::Vector3d::Zero();
	// cropped point cloud ptr by bounding box
	std::shared_ptr<open3d::geometry::PointCloud> pcd_cropped_ptr = std::make_shared<open3d::geometry::PointCloud>();
	
	// cd downsample will be done in the carSideJudgment function
	inline static double voxel_size_cd = 5.0;
	// icp downsaple will be done in the registration function
	inline static double voxel_size_icp = 2.5;
};

class __declspec(dllexport) CarLocSystem3D {

public:
	CarLocSystem3D() = default;
    ~CarLocSystem3D();

	/// @brief initialize the system
	/// @param data_dir given data directory
	/// @return initialization success or not
	bool init(std::string data_dir);



	/// @brief get robot modify pose observed by current camera
	Eigen::Vector6d calculateRobotModifyPose();

	/// @brief 判断车身左侧or右侧靠近相机
	/// @param online_left_pcd_ptr 左相机在线点云
	/// @param online_right_pcd_ptr 右相机在线点云
	/// @return 车身左侧or右侧靠近相机
	SIDE_ENUM carSideJudgment(const std::shared_ptr<open3d::geometry::PointCloud>& online_left_pcd_ptr, const std::shared_ptr<open3d::geometry::PointCloud>& online_right_pcd_ptr);

	/// @brief 计算车身到位偏差
	/// @param online_left_pcd_ptr 左相机在线点云
	/// @param online_right_pcd_ptr 右相机在线点云
	/// @return 计算成功or失败
	bool calculateCarPose(const std::shared_ptr<open3d::geometry::PointCloud>& online_left_pcd_ptr, const std::shared_ptr<open3d::geometry::PointCloud>& online_right_pcd_ptr);

public:	
	static std::string getSideString(const SIDE_ENUM& car_side);
	static SIDE_ENUM getSideEnum(const std::string& car_side_str);
	static std::string getPartString(const PART_ENUM& part_enum);
	static PART_ENUM getPartEnum(const std::string& part_str);

private:
	// internal functions
	
	static bool calculateCarPoseImpl(std::shared_ptr<open3d::geometry::PointCloud>& source_pcd_ptr,
		std::shared_ptr<open3d::geometry::PointCloud>& target_pcd_ptr,
		Eigen::Matrix4d& pose_target_source, bool vis_flag = false);
	
	/// @brief convert Eigen::Matrix4d to XYZWPR.
	/// @return pose in vector order: X Y Z in mm, W P R in degree.
	static Eigen::Vector6d toXYZWPR(const Eigen::Matrix4d& pose);

	/// @brief 6-dof pose optimization
	/// @param vpose input poses
	/// @return global optimized pose observed by all cameras
	static Eigen::Matrix4d jointlyOptimize(const std::vector<Eigen::Matrix4d>& vpose);
private:

	// init data
	std::unordered_map<SIDE_ENUM, std::unordered_map<PART_ENUM, PointCloudInfo>> _template_pcd_map;	// 储存左/右相机的前纵梁/后轮罩模板点云
	std::unordered_map<SIDE_ENUM, Eigen::Matrix4d> _pose_e2h_map;									// 储存左/右相机的眼在手外		

	SIDE_ENUM _car_side = SIDE_ENUM::Unknown;	// 车侧识别结果

	std::unordered_map<SIDE_ENUM, Eigen::Matrix4d> _pose_cam0_cam1_map;								// 储存左/右相机在线配准结果

	bool _reg_flag = false;

};

#endif // !CAR_LOC_SYSTEM_3D_H
