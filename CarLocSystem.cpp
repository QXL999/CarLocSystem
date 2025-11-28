#include "poseGraph3D.hpp"
#include "CarLocSystem3D.h"
#include "PointCloudRegistrationOpen3D.h"
#include "speedlog.h"
#include <open3d/Open3D.h>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <filesystem>
#include <future>
#include <algorithm>
namespace fs = std::filesystem;


Eigen::Matrix3d computeCovariance(std::shared_ptr<open3d::geometry::PointCloud>& pcd) {
	Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
	Eigen::Vector3d mean = pcd->GetCenter();

	for (const auto& point : pcd->points_) {
		Eigen::Vector3d centered = point - mean;
		cov += centered * centered.transpose();
	}

	return cov / pcd->points_.size();
}

bool tryInitialAlignment(
	std::shared_ptr<open3d::geometry::PointCloud>& source,
	std::shared_ptr<open3d::geometry::PointCloud>& target,
	Eigen::Matrix4d& pose) {

	LOG_INFO("Trying initial alignment...");

	// 使用PCA进行初始对齐
	auto source_down = source->VoxelDownSample(0.1);
	auto target_down = target->VoxelDownSample(0.1);

	if (!source_down || !target_down) {
		LOG_ERROR("Downsampling failed");
		return false;
	}

	// 计算PCA主方向
	Eigen::Matrix3d source_cov = computeCovariance(source_down);
	Eigen::Matrix3d target_cov = computeCovariance(target_down);

	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> source_eigen(source_cov);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> target_eigen(target_cov);

	Eigen::Matrix3d source_axes = source_eigen.eigenvectors();
	Eigen::Matrix3d target_axes = target_eigen.eigenvectors();

	// 构建初始变换
	pose.setIdentity();
	pose.block<3, 3>(0, 0) = target_axes * source_axes.transpose();

	// 计算初始平移（质心对齐）
	Eigen::Vector3d source_center = source_down->GetCenter();
	Eigen::Vector3d target_center = target_down->GetCenter();
	pose.block<3, 1>(0, 3) = target_center - pose.block<3, 3>(0, 0) * source_center;

	LOG_INFO("Initial alignment completed");
	return true;
}




// 计算点云边界框对角线长度
double calculateBoundingBoxDiagonal(std::shared_ptr<open3d::geometry::PointCloud>& pcd) 
{
	if (pcd->points_.empty()) return 1.0;

	Eigen::Vector3d min_bound = pcd->GetMinBound();
	Eigen::Vector3d max_bound = pcd->GetMaxBound();
	return (max_bound - min_bound).norm();
}

// 验证变换矩阵是否合理
bool isValidTransformation(const Eigen::Matrix4d& T) 
{
	// 检查平移量是否过大
	double translation_norm = T.block<3, 1>(0, 3).norm();
	if (translation_norm > 100.0) {  // 假设最大平移10米
		LOG_WARN("Translation too large: {:.3f}mm", translation_norm);
		return false;
	}

	// 检查旋转矩阵是否正交
	Eigen::Matrix3d R = T.block<3, 3>(0, 0);
	Eigen::Matrix3d I = R * R.transpose();
	double ortho_error = (I - Eigen::Matrix3d::Identity()).norm();
	if (ortho_error > 1e-3) {
		LOG_WARN("Rotation matrix not orthogonal: {:.6f}", ortho_error);
		return false;
	}

	// 检查行列式（反射检测）
	double det = R.determinant();
	if (std::abs(det - 1.0) > 1e-3) {
		LOG_WARN("Invalid rotation determinant: {:.6f}", det);
		return false;
	}

	return true;
}

// 备用配准方案
bool tryFallbackRegistration(
std::shared_ptr<open3d::geometry::PointCloud>& source,
std::shared_ptr<open3d::geometry::PointCloud>& target,
Eigen::Matrix4d& pose) {

	LOG_INFO("Trying fallback registration...");

	// 方案1：使用更宽松的参数
	PointCloudRegistrationOpen3D::PreprocessOptions preprocess_options;
	preprocess_options.voxel_size = 0.1;  // 更大的体素
	preprocess_options.filter_flag = true;
	preprocess_options.nb_points = 30;
	preprocess_options.search_radius = 0.5;

	PointCloudRegistrationOpen3D::RegistrationOptions reg_options;
	reg_options.voxel_size = preprocess_options.voxel_size;
	reg_options.distance_threshold_fgr = 0.2;  // 更大的距离阈值
	reg_options.distance_threshold_icp = 0.3;
	reg_options.iteration_num = 30;
	reg_options.tuple_test = false;  // 禁用tuple test
	reg_options.th_icp_pair_num = 200;

	bool succ = PointCloudRegistrationOpen3D::PCDRegistration(
		source, preprocess_options, target, preprocess_options,
		reg_options, pose, false);

	if (succ) {
		LOG_INFO("Fallback registration succeeded");
			return true;
		}

	// 方案2：使用初始变换估计
	return tryInitialAlignment(source, target, pose);
}


bool CarLocSystem3D::calculateCarPoseImpl(std::shared_ptr<open3d::geometry::PointCloud>& source_pcd_ptr,
	std::shared_ptr<open3d::geometry::PointCloud>& target_pcd_ptr,
	Eigen::Matrix4d& pose_target_source,bool vis_flag) 
{
	/******************若配准失败，将在线点云阈值框大一点即可****************************/
	LOG_INFO("Calculate Online Car Pose...");
	if (source_pcd_ptr->points_.empty() || target_pcd_ptr->points_.empty()) 
	{
		LOG_ERROR("Point cloud is empty - Source: {}, Target: {}",source_pcd_ptr->points_.size(), target_pcd_ptr->points_.size());
		return false;
	}
	LOG_INFO("Point cloud stats - Source: {} points, Target: {} points",source_pcd_ptr->points_.size(), target_pcd_ptr->points_.size());

	// 2. 自适应参数设置
	PointCloudRegistrationOpen3D::PreprocessOptions preprocess_options;

	// 根据点云大小自适应设置体素大小
	double bbox_diag = calculateBoundingBoxDiagonal(target_pcd_ptr);
	preprocess_options.voxel_size = std::min(5., bbox_diag * 0.008); 

	preprocess_options.filter_flag = true;
	preprocess_options.nb_points = 100;//什么含义？
	preprocess_options.search_radius = preprocess_options.voxel_size * 8.0;
	preprocess_options.reverse_normals = false;

	// 3. 配准参数优化
	PointCloudRegistrationOpen3D::RegistrationOptions reg_options;
	reg_options.voxel_size = preprocess_options.voxel_size;

	// 自适应距离阈值
	reg_options.distance_threshold_fgr = preprocess_options.voxel_size * 1.5;  // 增大阈值
	reg_options.distance_threshold_icp = preprocess_options.voxel_size * 2.;  // 增大阈值

	reg_options.iteration_num = 50;  // 减少迭代次数
	reg_options.tuple_test = true;
	reg_options.th_icp_pair_num = 1000;  // 减少匹配对数量

	LOG_INFO("Using parameters - Voxel: {:.3f}, FGR Dist: {:.3f}, ICP Dist: {:.3f}",
		preprocess_options.voxel_size,
		reg_options.distance_threshold_fgr,
		reg_options.distance_threshold_icp);

	// 4. 执行配准
	bool succ = PointCloudRegistrationOpen3D::PCDRegistration(
		source_pcd_ptr, preprocess_options,
		target_pcd_ptr, preprocess_options,
		reg_options, pose_target_source, vis_flag);

	if (succ) {
		LOG_DEBUG("Calculate Online Car Pose Success!");

		// 5. 验证配准结果
		if (isValidTransformation(pose_target_source)) {
			return true;
		}
		else {
			LOG_WARN("Registration succeeded but transformation is invalid");
			return false;
		}
	}
	else {
		LOG_ERROR("Calculate Online Car Pose Failed!");

		return false;
		//// 6. 失败后尝试备用方案
		//return tryFallbackRegistration(source_pcd_ptr, target_pcd_ptr, pose_target_source);
	}
}




//bool CarLocSystem3D::calculateCarPoseImpl(std::shared_ptr<open3d::geometry::PointCloud>& source_pcd_ptr,
//	std::shared_ptr<open3d::geometry::PointCloud>& target_pcd_ptr, Eigen::Matrix4d& pose_target_source, bool vis_flag) {
//	// TODO: 参数开放出来
//	//点云预处理参数设置
//	PointCloudRegistrationOpen3D::PreprocessOptions preprocess_optins;
//  preprocess_optins.voxel_size = PointCloudInfo::voxel_size_icp;//体素
//	preprocess_optins.filter_flag = true;//滤波
//	preprocess_optins.nb_points = 100 / PointCloudInfo::voxel_size_icp;//用于配准的点数
//	preprocess_optins.search_radius = 10.;
//	preprocess_optins.reverse_normals = false;
//	//点云配准参数设置
//	PointCloudRegistrationOpen3D::RegistrationOptions reg_options;
//	reg_options.voxel_size = PointCloudInfo::voxel_size_icp;
//	reg_options.distance_threshold_fgr = PointCloudInfo::voxel_size_icp * 0.8;//FGR配准阈值max
//	reg_options.distance_threshold_icp = PointCloudInfo::voxel_size_icp * 2.0;//ICP配准阈值
//	reg_options.iteration_num = 100;//迭代次数
//	reg_options.tuple_test = true;//剔除误匹配
//	reg_options.th_icp_pair_num = 1000;//ICP匹配点数量min
//
//	//本地模板点云
//	/*target_pcd_ptr = target_pcd_ptr->VoxelDownSample(PointCloudInfo::voxel_size_cd);*/
//	LOG_INFO("Calculate Online Car Pose...");
//	bool succ = PointCloudRegistrationOpen3D::PCDRegistration(source_pcd_ptr, preprocess_optins, target_pcd_ptr, preprocess_optins, 
//															  reg_options, pose_target_source, vis_flag);
//	if (succ) {
//		LOG_DEBUG("Calculate Online Car Pose Success!");
//		return true;
//	}
//	else {
//		LOG_ERROR("Calculate Online Car Pose Failed!");
//		return false;
//	}
//}

Eigen::Vector6d CarLocSystem3D::toXYZWPR(const Eigen::Matrix4d& pose) {

	// https://doc.rc-visard.com/v21.07/en/pose_format_fanuc.html
	Eigen::Matrix3d Rot = pose.topLeftCorner(3, 3);
	Eigen::Quaterniond q = Eigen::Quaterniond(Rot);
	Eigen::Vector3d t = pose.topRightCorner(3, 1);

	double qx= q.x(), qy = q.y(), qz = q.z(), qw = q.w();

	double R = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)) / M_PI * 180;
	double P = asin(2 * (qw * qy - qz * qx)) / M_PI * 180;
	double W = atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)) / M_PI * 180;


	return Eigen::Vector6d(t(0), t(1), t(2), W, P, R);

		//If you want to recover R from WPR, use:
		//Eigen::Matrix3d R_wpr = ((Eigen::AngleAxisd(R / 180 * M_PI, Eigen::Vector3d::UnitZ()))
		//	* (Eigen::AngleAxisd(P / 180 * M_PI, Eigen::Vector3d::UnitY()))
		//	* (Eigen::AngleAxisd(W / 180 * M_PI, Eigen::Vector3d::UnitX()))).toRotationMatrix();
}
Eigen::Matrix4d CarLocSystem3D::fromXYZWPR(const Eigen::Vector6d& pose)
{
	double x = pose(0);
	double y = pose(1);
	double z = pose(2);

	double W_deg = pose(3);  // Roll  (X轴)
	double P_deg = pose(4);  // Pitch (Y轴)
	double R_deg = pose(5);  // Yaw   (Z轴)

	// 转换为弧度
	double W = W_deg * M_PI / 180.0;
	double P = P_deg * M_PI / 180.0;
	double R = R_deg * M_PI / 180.0;

	// 按顺序构造旋转矩阵：X (W) -> Y (P) -> Z (R)
	// 即：先绕 X 轴转 W，再绕 Y 轴转 P，最后绕 Z 轴转 R
	Eigen::Matrix3d R_x = Eigen::AngleAxisd(W, Eigen::Vector3d::UnitX()).toRotationMatrix();
	Eigen::Matrix3d R_y = Eigen::AngleAxisd(P, Eigen::Vector3d::UnitY()).toRotationMatrix();
	Eigen::Matrix3d R_z = Eigen::AngleAxisd(R, Eigen::Vector3d::UnitZ()).toRotationMatrix();

	// 最终旋转矩阵：R = Rz * Ry * Rx
	Eigen::Matrix3d R_total = R_z * R_y * R_x;

	// 构造 4x4 齐次变换矩阵
	Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

	// 设置旋转部分
	T.block<3, 3>(0, 0) = R_total;

	// 设置平移部分
	T.block<3, 1>(0, 3) = Eigen::Vector3d(x, y, z);

	// 最后一行已经是 [0, 0, 0, 1]（通过 Identity 初始化）

	return T;
}



Eigen::Matrix4d CarLocSystem3D::updateCarPose(const Eigen::Vector6d& pose)
{
	Eigen::Matrix4d DeltaCarPose_car1_car0 = fromXYZWPR(pose);
	Eigen::Matrix4d CarPose_car1_Base2 = DeltaCarPose_car1_car0* _pose_template_base2;
	return CarPose_car1_Base2;
}

Eigen::Vector6d CarLocSystem3D::jointlyOptimize(const std::vector<Eigen::Matrix4d>& vpose) 
{
	
	LOG_INFO("Starting Jointly Optimization for Car Inspection");

	/*if (vpose.size() == 1) {
		LOG_INFO("Input Pose Size = 1, Return empty numpy.");
		return{};
	}*/

	// calc average pose
	std::vector<Eigen::Vector3d> vecs;
	std::vector<Eigen::Quaterniond> quats;
	for (const auto& pose : vpose) {

		Eigen::Matrix3d R = pose.block<3, 3>(0, 0);
		Eigen::Vector3d t = pose.block<3, 1>(0, 3);
		vecs.push_back(pose.block<3, 1>(0, 3));
		quats.push_back(Eigen::Quaterniond(R));
	}

	Eigen::Vector3d averageVec = Eigen::Vector3d::Zero();
	for (const auto& vec : vecs) {
		averageVec += vec;
	}
	averageVec /= vecs.size();

	Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
	for (size_t i = 0; i < quats.size(); ++i) {
		Eigen::Vector4d qvec = quats[i].normalized().coeffs();
		A += qvec * qvec.transpose();
	}
	Eigen::Matrix4d eigenVectors = Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d>(A).eigenvectors();
	Eigen::Vector4d averageQvec = eigenVectors.col(3);

	Eigen::Matrix4d T_average = Eigen::Matrix4d::Identity();
	T_average.block<3, 3>(0, 0) = Eigen::Quaterniond(averageQvec).toRotationMatrix();
	T_average.block<3, 1>(0, 3) = averageVec;

	Eigen::Quaterniond q_average = Eigen::Quaterniond(averageQvec);
	Eigen::Vector3d t_average = averageVec;

	std::stringstream ss_avg;
	ss_avg << T_average;
	LOG_INFO("Use Average Pose as Initial Pose:\n{}", ss_avg.str());
	Eigen::Vector6d T_No_optimized_XYZWPR = toXYZWPR(T_average);
	LOG_DEBUG("Before BA：T_No_optimized_XYZWPR:\n{}", T_No_optimized_XYZWPR.transpose());
	std::cout << "Before BA：T_No_optimized_XYZWPR = " << T_No_optimized_XYZWPR.transpose() << std::endl;

	LOG_INFO("Constructing Problem");
	ceres::Problem problem;

	//设置信息矩阵
	Eigen::Vector6d info = Eigen::Vector6d::Zero();
	info[0] = info[1] = info[2] = 10.0;  // 旋转权重
	info[3] = info[4] = info[5] = 1.0;   // 平移权重
	LOG_INFO("1.ceres::Problem problem");
	for (const auto& pose : vpose) 
	{
		Eigen::Matrix3d R = pose.block<3, 3>(0, 0);
		Eigen::Vector3d t = pose.block<3, 1>(0, 3);

		// 设置残差函数
		ceres::CostFunction* cost_function =PosePriorErrorTerm::Create(Eigen::Quaterniond(R), t, info);
		LOG_INFO("2.ceres::CostFunction* cost_function");
		//设置残差快
		problem.AddResidualBlock(cost_function,
			new ceres::CauchyLoss(0.5),  // 使用较小的loss处理异常值
			q_average.coeffs().data(),
			t_average.data());
	}
	problem.SetParameterization(q_average.coeffs().data(),new ceres::EigenQuaternionParameterization());

	// 设置合理的优化选项
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_type = ceres::TRUST_REGION;
	options.minimizer_progress_to_stdout = true;  // 开启输出查看优化过程
	options.max_num_iterations = 80;
	options.function_tolerance = 1e-8;
	options.gradient_tolerance = 1e-10;
	options.parameter_tolerance = 1e-10;
	LOG_INFO("3.options.parameter_tolerance");
	/*LOG_INFO("Constructing Problem...");
	ceres::Problem problem;
	for (const auto& pose : vpose) {
		Eigen::Matrix3d R = pose.block<3, 3>(0, 0);
		ceres::CostFunction* cost_function = PosePriorErrorTerm::Create(Eigen::Quaterniond(R), pose.block<3, 1>(0, 3), Eigen::Vector6d::Identity());
		problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(1.0), q_average.coeffs().data(), t_average.data());
	}
	problem.SetParameterization(q_average.coeffs().data(), new ceres::EigenQuaternionParameterization());
	
	LOG_INFO("Starting Optimization...");
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_type = ceres::TRUST_REGION;
	options.minimizer_progress_to_stdout = false;
	options.max_num_iterations = 100;
	options.num_threads = 1;*/
	Eigen::Vector6d T_optimized_XYZWPR = Eigen::Vector6d::Zero();
	try
	{
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		std::stringstream ss_report;
		ss_report << summary.BriefReport();
		LOG_INFO(ss_report.str());
		LOG_INFO("Optimization Done!");

		Eigen::Matrix4d T_optimized = Eigen::Matrix4d::Identity();

		T_optimized.block<3, 3>(0, 0) = q_average.toRotationMatrix();
		T_optimized.block<3, 1>(0, 3) = t_average;
		std::stringstream ss;
		ss << T_optimized;
		//std::cout << "BA优化后 T_optimized: " << ss.str() << std::endl;
		LOG_DEBUG("Optimized Pose:\n{}", ss.str());
		//if (robotname == "SL")
		//{
		//	T_optimized = T_optimized.inverse();// 思灵  b1<-b0  
		//}
		T_optimized_XYZWPR = toXYZWPR(T_optimized);
		std::cout << "After BA_Algorithm：T_optimized_XYZWPR = " << T_optimized_XYZWPR.transpose() << std::endl;

		LOG_DEBUG("After BA_Algorithm：T_optimized_XYZWPR:\n{}", T_optimized_XYZWPR.transpose());
		if (T_optimized_XYZWPR[3] >= 2 || T_optimized_XYZWPR[4] >= 2 || T_optimized_XYZWPR[5] >= 2)
		{
			std::cout << "BA优化后的结果存在问题，采用初始值！" << std::endl;
			LOG_INFO("The result after BA optimization has issues; use the initial value!", ss.str());
			return T_No_optimized_XYZWPR;
		}
	}
	catch (const std::exception& e)
	{
		LOG_ERROR("Optimization crashed: {}", e.what());
		LOG_WARN("Reset optimization result to identity pose");
	}

	return T_optimized_XYZWPR;
}

//CarLocSystem3D::~CarLocSystem3D() 
//{
//	std::cout << "析构对象：" << robotname << std::endl;
//	SpeedLogger::getInstance()->unInit();
//}

bool CarLocSystem3D::StaticInitLog()
{
	SpeedLogger::LoggerConfig logConfig("./Log_CarLocSystem/", "XM");
	SpeedLogger::getInstance()->init(logConfig);
	LOG_INFO("Log Init Success!");
	return true;
}
bool CarLocSystem3D::DestructionLog()
{
	LOG_WARN("Log will be unInit!");
	SpeedLogger::getInstance()->unInit();
	return true;
}

bool CarLocSystem3D::init(std::string data_dir, std::string robot_name) 
{
	//robotname = robot_name;
	if (robot_name == "SL")
	{
		SpeedLogger::LoggerConfig logConfig("./Log_CarLocSystem/", "SL");
		SpeedLogger::getInstance()->init(logConfig);

		// Step 1 加载零偏模板点云
		std::string templates_dir = data_dir + "\\Templates";
		if (!fs::exists(templates_dir) || !fs::is_directory(templates_dir)) {
			LOG_ERROR("Not Found {}", templates_dir);
			return false;
		}

		for (const auto& cam_dir : fs::directory_iterator(templates_dir)) {
			if (!fs::is_directory(cam_dir))
				continue;

			// Step 1.1: Load One Side Template Point Cloud
			std::string cam_side_str = cam_dir.path().filename().string();
			SIDE_ENUM cam_side_enum = CarLocSystem3D::getSideEnum(cam_side_str);
			LOG_INFO("Step 1: Loading {} Side Camera Template PointCloud...", cam_side_str);

			std::unordered_map<PART_ENUM, PointCloudInfo> part_pcd_map;
			for (const auto& part_dir : fs::directory_iterator(cam_dir)) {
				if (!fs::is_directory(part_dir))
					continue;

				std::string part_name = part_dir.path().filename().string();
				PART_ENUM part_enum = CarLocSystem3D::getPartEnum(part_name);

				LOG_INFO("Loading {} Side {} Template Point Cloud...", cam_side_str, part_name);
				std::string path_template_pcd = part_dir.path().string() + R"(\template.ply)";
				std::shared_ptr<open3d::geometry::PointCloud> template_pcd_ptr = std::make_shared<open3d::geometry::PointCloud>();
				//open3d::io::ReadPointCloudOption option = { "auto", true, true };
				//open3d::io::ReadPointCloud(path_template_pcd, *template_pcd_ptr, option);
				open3d::io::ReadPointCloud(path_template_pcd, *template_pcd_ptr);
				if (template_pcd_ptr->IsEmpty()) 
				{
					LOG_ERROR("Failed to Load {}", path_template_pcd);
					return false;
				}

				// Step 1.2: Crop Template Point Cloud
				std::string roi_path = part_dir.path().string() + "\\ROI.yml";
				LOG_INFO("Trying to Load ROI -> {}", roi_path);
				cv::FileStorage fs(roi_path, cv::FileStorage::READ);
				if (!fs.isOpened()) {
					LOG_ERROR("Failed to Load ROI.");
					return false;
				}
				double min_x, min_y, min_z, max_x, max_y, max_z;
				fs["min_bound_x"] >> min_x;
				fs["min_bound_y"] >> min_y;
				fs["min_bound_z"] >> min_z;
				fs["max_bound_x"] >> max_x;
				fs["max_bound_y"] >> max_y;
				fs["max_bound_z"] >> max_z;
				fs.release();
				LOG_INFO("ROI Loaded.");

				PointCloudInfo template_pcd_info;
				// scale 
				template_pcd_info.pcd_ptr = template_pcd_ptr;
				template_pcd_info.min_bound = Eigen::Vector3d(min_x, min_y, min_z);
				template_pcd_info.max_bound = Eigen::Vector3d(max_x, max_y, max_z);
				template_pcd_info.pcd_cropped_ptr = template_pcd_info.pcd_ptr->Crop(open3d::geometry::AxisAlignedBoundingBox(
					template_pcd_info.min_bound, template_pcd_info.max_bound));
				part_pcd_map.insert(std::make_pair(part_enum, template_pcd_info));
				LOG_INFO("{} Side {} Template Point Cloud Loaded.", cam_side_str, part_name);

				continue;
				{
					open3d::visualization::Visualizer vis;
					vis.CreateVisualizerWindow("Template", 1920, 1080);
					auto base_axis_ptr = open3d::geometry::TriangleMesh::CreateCoordinateFrame(500);
					vis.AddGeometry(base_axis_ptr);
					vis.AddGeometry(template_pcd_info.pcd_cropped_ptr);
					vis.Run();
				}
			}

			this->_template_pcd_map.insert(std::make_pair(cam_side_enum, part_pcd_map));


			// step 2 加载眼在手外标定文件
			LOG_INFO("Step 2: Loading {} Side Camera Calibration.yml...", getSideString(cam_side_enum));
			std::string path_e2h = cam_dir.path().string() + "\\Calibration.yml";
			cv::FileStorage fs_in(path_e2h, cv::FileStorage::READ);
			if (!fs_in.isOpened()) {
				LOG_ERROR("Failed to Open {}", path_e2h);
				return false;
			}
			cv::Mat data;
			fs_in["new_extrinsics"] >> data;
			Eigen::Map<Eigen::Vector3d> t(data.ptr<double>(0));
			Eigen::Map<Eigen::Quaterniond> q(data.ptr<double>(0) + 3);
			Eigen::Matrix4d pose_e2h = Eigen::Matrix4d::Identity();
			pose_e2h.topLeftCorner(3, 3) = q.toRotationMatrix();
			pose_e2h.topRightCorner(3, 1) = t;
			fs_in.release();

			this->_pose_e2h_map.insert(std::make_pair(cam_side_enum, pose_e2h));

			LOG_INFO("Load {} Side Done!\n", getSideString(cam_side_enum));
		}

		// Check
		LOG_INFO("Checking Templates...");
		for (const auto& [side_enum, part_pcd_map] : this->_template_pcd_map) {
			LOG_INFO("{} Side Camera Info:", getSideString(side_enum));
			for (const auto& [part_enum, pcd_info] : part_pcd_map) {
				LOG_INFO("--{} PCD Crop {} -> {}", getPartString(part_enum), pcd_info.pcd_ptr->points_.size(), pcd_info.pcd_cropped_ptr->points_.size());
			}
		}
		LOG_INFO("Checking e2h...");
		for (const auto& [side_enum, pose_e2h] : this->_pose_e2h_map) {
			std::stringstream ss;
			ss << pose_e2h;
			LOG_INFO("{} Side Camera e2h:\n{}", getSideString(side_enum), ss.str());
		}

		return true;

		{
			open3d::visualization::Visualizer vis;
			vis.CreateVisualizerWindow("Template", 1920, 1080);
			auto base_axis_ptr = open3d::geometry::TriangleMesh::CreateCoordinateFrame(500);
			vis.AddGeometry(base_axis_ptr);
			for (const auto& [side_enum, pose_e2h] : this->_pose_e2h_map) {
				auto cam_axis_ptr = open3d::geometry::TriangleMesh::CreateCoordinateFrame(200);
				cam_axis_ptr->Transform(pose_e2h);
				vis.AddGeometry(cam_axis_ptr);
			}
			vis.Run();
		}
	}
	if(robot_name=="R1"||robot_name=="R2"||robot_name=="R3"||robot_name=="R4")
	{

		//_robot_name = robot_name;

		// step 1 加载CAD点云与零偏实拍点云
		LOG_INFO("Step 1: Loading CAD PointCloud and Zero PointCloud...");
		LOG_INFO("Loading pcd_under_model.pcd...");
		_cad_pcd_ptr = std::make_shared<open3d::geometry::PointCloud>();
		if (!open3d::io::ReadPointCloudFromPCD(data_dir + "\\pcd_under_model.pcd", *_cad_pcd_ptr, open3d::io::ReadPointCloudOption())) {
			LOG_ERROR("Failed to load {}", data_dir + "\\pcd_under_model.pcd");
			return false;
		}
		LOG_INFO("Done!");
		//现场3D相机实拍点云
		LOG_INFO("Loading template.ply...");
		_cam0_pcd_ptr = std::make_shared<open3d::geometry::PointCloud>();
		open3d::io::ReadPointCloud(data_dir + "\\template.ply", *_cam0_pcd_ptr, open3d::io::ReadPointCloudOption());
		if (_cam0_pcd_ptr->IsEmpty()) 
		{
			LOG_ERROR("Failed to load {}", data_dir + "\\template.ply");
			return false;
		}
		//小米模板点云切割！
		LOG_INFO("XM templatePCD ROI Crop...");
		std::string roi_path = data_dir + "\\ROI.yml";
		LOG_INFO("Trying to Load ROI -> {}", roi_path);
		cv::FileStorage fs(roi_path, cv::FileStorage::READ);
		if (!fs.isOpened()) {
			LOG_ERROR("Failed to Load ROI.");
			return false;
		}
		double min_x, min_y, min_z, max_x, max_y, max_z;
		fs["min_bound_x"] >> min_x;
		fs["min_bound_y"] >> min_y;
		fs["min_bound_z"] >> min_z;
		fs["max_bound_x"] >> max_x;
		fs["max_bound_y"] >> max_y;
		fs["max_bound_z"] >> max_z;
		fs.release();
		LOG_INFO("ROI Loaded.");
		const Eigen::Vector3d min_bound = { min_x,min_y,min_z };
		const Eigen::Vector3d max_bound = { max_x,max_y,max_z };
		ROI.min_bound = min_bound;
        ROI.max_bound = max_bound;
		_cam0_pcd_ptr = _cam0_pcd_ptr->Crop(open3d::geometry::AxisAlignedBoundingBox(min_bound, max_bound));
		//_cam0_pcd_ptr = _cam0_pcd_ptr->VoxelDownSample(PointCloudInfo::voxel_size_cd - 2);
		LOG_INFO("_cam0_pcd_ptr ROI Cropped.");
		LOG_INFO("Done!");

		// step 2 加载眼在手外标定文件
		LOG_INFO("Step 2: Loading Calibration.yml...");
		cv::FileStorage fs_in(data_dir + "\\Calibration.yml", cv::FileStorage::READ);
		if (!fs_in.isOpened()) {
			LOG_ERROR("Failed to Open {}", data_dir + "\\Calibration.yml");
			return false;
		}
		cv::Mat data;
		fs_in["new_extrinsics"] >> data;
		Eigen::Map<Eigen::Vector3d> t(data.ptr<double>(0));
		Eigen::Map<Eigen::Quaterniond> q(data.ptr<double>(0) + 3);
		_pose_bc.topLeftCorner(3, 3) = q.toRotationMatrix();
		_pose_bc.topRightCorner(3, 1) = t;
		LOG_INFO("Done!");
		fs_in.release();


		//step 3 加载模板点云相对base的标定文件

		LOG_INFO("Step 3: Load pose_TemplatePCD_Base ...");
		std::string path_pose_template_pcd_base2 = data_dir + "\\Pose_Yu7_Robot.txt";
		std::ifstream file(path_pose_template_pcd_base2);
		if (!file.is_open())
		{
			LOG_ERROR("File Pose_Yu7_Robot.txt is empty or dictory error ...");
			std::cerr << "无法打开文件: " << path_pose_template_pcd_base2 << std::endl;
			return false;
		}
		std::string line;
		double x, y, z, qx, qy, qz, qw;
		while (std::getline(file, line))
		{
			std::istringstream iss(line);
			if (!(iss >> x >> y >> z >> qx >> qy >> qz >> qw)) 
			{
				std::cerr << "Pose_Yu7_Robot.txt文件内容错误！" << std::endl;
				return false;
			}
		}
		_pose_template_base2.topLeftCorner(3, 3) = Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();
		_pose_template_base2.topRightCorner(3, 1) = Eigen::Vector3d(x, y, z);
	}
	// step 3 零偏实拍与CAD点云配准
	LOG_INFO("Step 3: Load / Calculate T_cad_c0...");
	// if pose_cad_cam0.yml exists, load it. otherwise, calculate it.
	std::string path_pose_cad_cam0 = data_dir + "\\pose_cad_cam0.yml";
	if (!std::filesystem::exists(path_pose_cad_cam0)) {
		LOG_WARN("Not Found {}, Calling Registration...", path_pose_cad_cam0);

		if (!calculateCarPoseImpl(_cam0_pcd_ptr, _cad_pcd_ptr, _pose_cad_cam0)) {
			LOG_ERROR("Failed to Calculate T_cad_c0!");
			return false;
		}

		// write _pose_cad_cam0 to file
		Eigen::Matrix3d R_cad_cam0 = _pose_cad_cam0.block<3, 3>(0, 0);
		Eigen::Quaterniond q_cad_cam0(R_cad_cam0);
		Eigen::Vector3d t_cad_cam0(_pose_cad_cam0.block<3, 1>(0, 3));
		cv::Mat xyzq = (cv::Mat_<double>(1, 7) << t_cad_cam0[0], t_cad_cam0[1], t_cad_cam0[2], q_cad_cam0.x(), q_cad_cam0.y(), q_cad_cam0.z(), q_cad_cam0.w());
		cv::FileStorage fs_out(path_pose_cad_cam0, cv::FileStorage::WRITE);
		fs_out << "pose_cad_cam0" << xyzq;
		fs_out.release();

		LOG_INFO("Done!");
		LOG_INFO("Init System Done!\n");
		return true;
	}
	else {
		LOG_WARN("Found {}, Loading...", path_pose_cad_cam0);
		cv::FileStorage fs_in(path_pose_cad_cam0, cv::FileStorage::READ);
		if (!fs_in.isOpened()) {
			LOG_ERROR("Failed to Open {}", path_pose_cad_cam0);
			return false;
		}
		cv::Mat data;
		fs_in["pose_cad_cam0"] >> data;
		Eigen::Map<Eigen::Vector3d> t(data.ptr<double>(0));
		Eigen::Map<Eigen::Quaterniond> q(data.ptr<double>(0) + 3);
		_pose_cad_cam0.topLeftCorner(3, 3) = q.toRotationMatrix();
		_pose_cad_cam0.topRightCorner(3, 1) = t;
		LOG_INFO("Done!");
		fs_in.release();
		return true;
	}
	
	LOG_ERROR("Incorrect robotname parameter input. Initialization failed！");
	return false;
}


Eigen::Vector6d CarLocSystem3D::calculateRobotModifyPose(std::string robot_name)
{
	if (robot_name == "SL")
	{
		LOG_INFO("calRobotModifyPose_SL");
		// check status
		if (this->_car_side == SIDE_ENUM::Unknown || !this->_reg_flag_sl || this->_pose_e2h_map.empty()) {
			LOG_ERROR("calculateRobotModifyPose_SL():Wrong System Status, Return Eigen::Vector6d::Zero() Instead!");
			return Eigen::Vector6d::Zero();
		}

		// pose_b0b1 = _pose_bc * _pose_cam0_cam1 * _pose_bc.inverse();
		Eigen::Matrix4d delta_pose_bb_left = _pose_e2h_map[SIDE_ENUM::Left] * _pose_cam0_cam1_map[SIDE_ENUM::Left] * _pose_e2h_map[SIDE_ENUM::Left].inverse();
		Eigen::Matrix4d delta_pose_bb_right = _pose_e2h_map[SIDE_ENUM::Right] * _pose_cam0_cam1_map[SIDE_ENUM::Right] * _pose_e2h_map[SIDE_ENUM::Right].inverse();
		
		Eigen::Vector6d delta_pose_bb_XYZWPR_left = toXYZWPR(delta_pose_bb_left.inverse());
        Eigen::Vector6d delta_pose_bb_XYZWPR_right = toXYZWPR(delta_pose_bb_right.inverse());
		std::cout<<"delta_pose_bb_XYZWPR_left: "<<delta_pose_bb_XYZWPR_left.transpose()<<std::endl;
        std::cout<<"delta_pose_bb_XYZWPR_right: "<<delta_pose_bb_XYZWPR_right.transpose()<<std::endl;

		//思灵只有一台机器人，故可以计算均值；小米则不可以
		Eigen::Vector6d delta_pose_bb_XYZWPR = jointlyOptimize({ delta_pose_bb_left, delta_pose_bb_right });

	/*	Eigen::Vector6d delta_XYZWPR = toXYZWPR(delta_pose_bb);*/
		LOG_INFO("SL_delta_XYZWPR: {}", delta_pose_bb_XYZWPR.transpose());
		LOG_INFO("getRobotModifyPose_SL");
		return delta_pose_bb_XYZWPR;
	}
	if (robot_name == "R1" || robot_name == "R2" || robot_name == "R3" || robot_name == "R4")
	{
		LOG_INFO("calRobotModifyPose_XM");
		// return pose_b0_b1,从base1->base0
		if (!_reg_flag_xm)
		{
			LOG_ERROR("calculateRobotModifyPose_XM():Calculate Car Pose Failed, Return Eigen::Matrix4d::Zero() Instead!");
			return Eigen::Vector6d::Zero();
		}

		Eigen::Matrix4d pose_b0b1 = _pose_bc * _pose_cam0_cam1 * _pose_bc.inverse();
		Eigen::Matrix4d pose_b1b0 = pose_b0b1.inverse();//从b0变换到b1
		std::stringstream ss;
		ss << pose_b1b0;
		LOG_DEBUG("Return Robot Modify Pose_XM， pose_b1b0:\n{}", ss.str());
		Eigen::Vector6d delta_XYZWPR = toXYZWPR(pose_b1b0);
		LOG_INFO("XM_delta_XYZWPR: {}", delta_XYZWPR.transpose());
		LOG_INFO("getRobotModifyPose_XM");
		return delta_XYZWPR;
	}
	LOG_ERROR("Incorrect robotname parameter input. Initialization failed！");
	return Eigen::Vector6d::Zero();
}

//Eigen::Vector6d CarLocSystem3D::calculateRobotModifyPose_XM() 
//{
//	LOG_INFO("calRobotModifyPose_XM");
//	// return pose_b0_b1,从base1->base0
//	if (!_reg_flag_xm)
//	{
//		LOG_ERROR("calculateRobotModifyPose_XM():Calculate Car Pose Failed, Return Eigen::Matrix4d::Zero() Instead!");
//		return Eigen::Vector6d::Zero();
//	}
//
//	Eigen::Matrix4d pose_b0b1 = _pose_bc * _pose_cam0_cam1 * _pose_bc.inverse();
//
//	std::stringstream ss;
//	ss << pose_b0b1;
//	LOG_DEBUG("Return Robot Modify Pose_XM:\n{}", ss.str());
//	Eigen::Vector6d delta_XYZWPR = toXYZWPR(pose_b0b1);
//	LOG_INFO("XM_delta_XYZWPR: {}", delta_XYZWPR.transpose());
//	LOG_INFO("getRobotModifyPose_XM");
//	return delta_XYZWPR;
//}

SIDE_ENUM CarLocSystem3D::carSideJudgment(const std::shared_ptr<open3d::geometry::PointCloud>& online_left_pcd_ptr, 
										  const std::shared_ptr<open3d::geometry::PointCloud>& online_right_pcd_ptr) {
	LOG_INFO("carSideJudgment()+");

	// check
	if (!online_left_pcd_ptr || !online_right_pcd_ptr) {
		LOG_ERROR("online_left_pcd_ptr or online_right_pcd_ptr is nullptr!");
		return SIDE_ENUM::Unknown;
	}
	if (online_left_pcd_ptr->points_.size() == 0 || online_right_pcd_ptr->points_.size() == 0) {
		LOG_ERROR("online_left_pcd_ptr or online_right_pcd_ptr points is empty!");
		return SIDE_ENUM::Unknown;
	}

	this->_car_side = SIDE_ENUM::Unknown;	// reset _car_side
	auto t1 = std::chrono::high_resolution_clock::now();

	std::unordered_map<SIDE_ENUM, std::shared_ptr<open3d::geometry::PointCloud>> online_pcd_map;
	online_pcd_map.insert(std::make_pair(SIDE_ENUM::Left, online_left_pcd_ptr));
	online_pcd_map.insert(std::make_pair(SIDE_ENUM::Right, online_right_pcd_ptr));
	
	// 用于决定最后的结果 例如左边的哪一个部分
	std::unordered_map<SIDE_ENUM, PART_ENUM> side_part_map;
	for (const auto& [cam_side_enum, online_pcd_ptr] : online_pcd_map) {
	/*	auto online_down_pcd_ptr = online_pcd_ptr->VoxelDownSample(PointCloudInfo::voxel_size_cd);*/
		auto online_down_pcd_ptr = online_pcd_ptr;
		std::unordered_map<PART_ENUM, double> _carSide_dist_map;

		// 左/右相机模板点云
		auto& template_side_pcd_map = this->_template_pcd_map[cam_side_enum];
		for (const auto& [part_enum, template_pcd_info] : template_side_pcd_map) {
			LOG_INFO("{} Side Camera", getSideString(cam_side_enum));
			//本地模板点云
			auto template_final_pcd_ptr = template_pcd_info.pcd_cropped_ptr->VoxelDownSample(PointCloudInfo::voxel_size_cd);
			//在线采集点云
			auto online_final_pcd_ptr = online_down_pcd_ptr->Crop(open3d::geometry::AxisAlignedBoundingBox(template_pcd_info.min_bound, template_pcd_info.max_bound));

			LOG_INFO("\tDownSample And Cropped Template ply Size: {}", getPartString(part_enum), template_final_pcd_ptr->points_.size());
			LOG_INFO("\tDownSample And Cropped Online PCD Size: {}", online_final_pcd_ptr->points_.size());

			if (online_final_pcd_ptr->points_.size() == 0) 
			{
				// 在线点云按照模版点云的Bounding Box裁剪后为空，说明不匹配
				LOG_INFO("\tCD Distance: {}", std::numeric_limits<double>::max());
				_carSide_dist_map.emplace(part_enum, std::numeric_limits<double>::max());
			}
			else {
				// Symmetric Chamfer Distance
				auto vdist_AB = online_final_pcd_ptr->ComputePointCloudDistance(*template_final_pcd_ptr);
				auto vdist_BA = template_final_pcd_ptr->ComputePointCloudDistance(*online_final_pcd_ptr);
				double distAB = 0.0, distBA = 0.0;
				//平均距离，同一方向的在线点云一组与本地的两组点云进行配准
				distAB = std::accumulate(vdist_AB.begin(), vdist_AB.end(), distAB);
				distAB /= double(vdist_AB.size());
				distBA = std::accumulate(vdist_BA.begin(), vdist_BA.end(), distBA);
				distBA /= double(vdist_BA.size());

				double cham = distAB + distBA;
				LOG_INFO("\tCD Distance: {}", cham);

				_carSide_dist_map.emplace(part_enum, cham);

				continue;
				{
					open3d::visualization::Visualizer vis;
					std::string win_name = "CD Distance: " + std::to_string(cham);
					vis.CreateVisualizerWindow(win_name, 1920, 1080);
					auto base_axis_ptr = open3d::geometry::TriangleMesh::CreateCoordinateFrame(200);
					template_final_pcd_ptr->PaintUniformColor(Eigen::Vector3d(1., 0., 0.)); // Red
					online_final_pcd_ptr->PaintUniformColor(Eigen::Vector3d(0., 1., 0.));
					vis.AddGeometry(base_axis_ptr);
					vis.AddGeometry(template_final_pcd_ptr);
					vis.AddGeometry(online_final_pcd_ptr);
					vis.Run();
				}
				
			}

		}

		double min_dist = std::numeric_limits<double>::max();
		for (const auto& pair : _carSide_dist_map) {
			if (pair.second < min_dist) {
				min_dist = pair.second;		// 更新最小值
				side_part_map[cam_side_enum] = pair.first;	// 更新对应的 part
			}
		}


	}
	LOG_INFO("carSideJudgment()-");
	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() * 0.001;
	LOG_INFO("Time Cost: {}s", duration);

	if (side_part_map[SIDE_ENUM::Left] == PART_ENUM::FrontLongitudinalBeam && side_part_map[SIDE_ENUM::Right] == PART_ENUM::RearWheelHouse) {
		this->_car_side = SIDE_ENUM::Left;//车的方向
		LOG_INFO("Car Side Result: Left");
		return this->_car_side;
	}
	if (side_part_map[SIDE_ENUM::Left] == PART_ENUM::RearWheelHouse && side_part_map[SIDE_ENUM::Right] == PART_ENUM::FrontLongitudinalBeam) {
		this->_car_side = SIDE_ENUM::Right;
		LOG_INFO("Car Side Result: Right");
		return this->_car_side;
	}
	LOG_ERROR("Car Side Result: Unknown");
	return SIDE_ENUM::Unknown;
}

bool CarLocSystem3D::calculateCarPose(const std::shared_ptr<open3d::geometry::PointCloud>& online_left_pcd_ptr, 
									  const std::shared_ptr<open3d::geometry::PointCloud>& online_right_pcd_ptr) 
{
	
	if (online_right_pcd_ptr->IsEmpty())
	{
		//XM
		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

		LOG_INFO("XM Calculate Online Car Pose...");
		bool vis_flag = false;
		//_pose_cam0_cam1为相机坐标系下，车身位姿的改变量,cam1_pcd_ptr在线点云，应该是1变换到0
		//_cam0_pcd_ptr = _cam0_pcd_ptr->VoxelDownSample(PointCloudInfo::voxel_size_cd-3.);

		auto online_left_pcd_ptr_Crop = online_left_pcd_ptr->Crop(open3d::geometry::AxisAlignedBoundingBox(ROI.min_bound, ROI.max_bound + Eigen::Vector3d(50, 50, 100)));
		bool succ = calculateCarPoseImpl(online_left_pcd_ptr_Crop, _cam0_pcd_ptr, _pose_cam0_cam1, vis_flag);

		std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
		std::cout<<"Time Cost: "<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() * 0.001<<"s."<<std::endl;
		LOG_INFO("Time Cost: {}s.", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() * 0.001);

		if (succ) {
			_reg_flag_xm = true;
			LOG_DEBUG("XM Calculate Online Car Pose Success!");
			Eigen::Vector6d delta_XYZWPR = CarLocSystem3D::toXYZWPR(_pose_cam0_cam1);
			std::cout << "XM _pose_cam0_cam1: " << delta_XYZWPR.transpose() << std::endl;
			return true;
		}
		else {
			_reg_flag_xm = false;
			LOG_ERROR("XM Calculate Online Car Pose Failed!");
			return false;
		}
	}
	else
	{
		//SL
		LOG_INFO("SL calculateCarPose()+");
		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

		// check
		if (this->_car_side == SIDE_ENUM::Unknown) {
			LOG_ERROR("You Must Calculate Car Side Before Calling calculateCarPose()!");
			return false;
		}
		// reset
		this->_pose_cam0_cam1_map[SIDE_ENUM::Left] = Eigen::Matrix4d::Identity();
		this->_pose_cam0_cam1_map[SIDE_ENUM::Right] = Eigen::Matrix4d::Identity();
		this->_reg_flag_sl = false;


		// Step 0: Get Ref Template PCD for Left and Right Camera
		// Ref Template Info
		std::shared_ptr<PointCloudInfo> left_ref_pcd_info = std::make_shared<PointCloudInfo>();
		std::shared_ptr<PointCloudInfo> right_ref_pcd_info = std::make_shared<PointCloudInfo>();

		switch (this->_car_side) {

			// 车身左侧靠近3D相机，
		case SIDE_ENUM::Left:
			// 左相机参考前纵梁点云，右相机参考后轮罩点云
			*left_ref_pcd_info = this->_template_pcd_map[SIDE_ENUM::Left][PART_ENUM::FrontLongitudinalBeam];
			*right_ref_pcd_info = this->_template_pcd_map[SIDE_ENUM::Right][PART_ENUM::RearWheelHouse];
			break;
			// 车身右侧靠近3D相机，
		case SIDE_ENUM::Right:
			// 左相机参考后轮罩点云，右相机参考前纵梁点云
			*left_ref_pcd_info = this->_template_pcd_map[SIDE_ENUM::Left][PART_ENUM::RearWheelHouse];
			*right_ref_pcd_info = this->_template_pcd_map[SIDE_ENUM::Right][PART_ENUM::FrontLongitudinalBeam];
			break;
		}

		// Step 1: Get Cropped PCD
		// Cropped Template PCD
		auto left_ref_cropped_pcd_ptr = left_ref_pcd_info->pcd_cropped_ptr;
		std::cout<<"left_ref_cropped_pcd_ptr "<<left_ref_cropped_pcd_ptr->points_.size()<<std::endl;
		auto right_ref_cropped_pcd_ptr = right_ref_pcd_info->pcd_cropped_ptr;
        std::cout<<"right_ref_cropped_pcd_ptr "<<right_ref_cropped_pcd_ptr->points_.size()<<std::endl;

		// Cropped Online PCD
		// 左相机在线点云按照前纵梁/后轮罩裁剪，右相机在线点云按照后轮罩/前纵梁裁剪
		auto left_online_cropped_pcd_ptr = online_left_pcd_ptr->Crop(open3d::geometry::AxisAlignedBoundingBox(left_ref_pcd_info->min_bound- Eigen::Vector3d(20,20,20), left_ref_pcd_info->max_bound));
		auto right_online_cropped_pcd_ptr = online_right_pcd_ptr->Crop(open3d::geometry::AxisAlignedBoundingBox(right_ref_pcd_info->min_bound - Eigen::Vector3d(20, 20, 20), right_ref_pcd_info->max_bound));
		std::cout<<"left_online_cropped_pcd_ptr "<<left_online_cropped_pcd_ptr->points_.size()<<std::endl;
        std::cout<<"right_online_cropped_pcd_ptr "<<right_online_cropped_pcd_ptr->points_.size()<<std::endl;

		// Step 2: Call Registration

		bool vis_flag = false;
		// 多线程
		auto left_future = std::async(std::launch::async, calculateCarPoseImpl,
			std::ref(left_online_cropped_pcd_ptr), std::ref(left_ref_cropped_pcd_ptr),
			std::ref(this->_pose_cam0_cam1_map[SIDE_ENUM::Left]), vis_flag);
		auto right_future = std::async(std::launch::async, calculateCarPoseImpl,
			std::ref(right_online_cropped_pcd_ptr), std::ref(right_ref_cropped_pcd_ptr),
			std::ref(this->_pose_cam0_cam1_map[SIDE_ENUM::Right]), vis_flag);
		//保证线程执行完毕，返回calculateCarPoseImpl的值
		bool left_succ = left_future.get();
		std::cout<<"left_succ "<<left_succ<<std::endl;
		bool right_succ = right_future.get();
        std::cout<<"right_succ "<<right_succ<<std::endl;
		_reg_flag_sl = left_succ && right_succ;


		//_reg_flag_sl = calculateCarPoseImpl(left_online_cropped_pcd_ptr, left_ref_cropped_pcd_ptr, this->_pose_cam0_cam1_map[SIDE_ENUM::Left], vis_flag) &&
		//    		calculateCarPoseImpl(right_online_cropped_pcd_ptr, right_ref_cropped_pcd_ptr, this->_pose_cam0_cam1_map[SIDE_ENUM::Right], vis_flag);

		_reg_flag_sl ? LOG_INFO("SL Registration Success!") : LOG_ERROR("SL Registration Failed!");
		LOG_INFO("SL calculateCarPose()-");
		std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
		LOG_INFO("SL Time Cost: {}s.", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() * 0.001);

		return _reg_flag_sl;
	}
	return false;

}

//bool CarLocSystem3D::calculateCarPose(std::shared_ptr<open3d::geometry::PointCloud> cam1_pcd_ptr) {
//	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
//
//	LOG_INFO("Calculate Online Car Pose...");
//	bool vis_flag = false;
//	//_pose_cam0_cam1为相机坐标系下，车身位姿的改变量,cam1_pcd_ptr在线点云，应该是1变换到0
//	bool succ = calculateCarPoseImpl(cam1_pcd_ptr, _cam0_pcd_ptr, _pose_cam0_cam1, vis_flag);
//
//	std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//	LOG_INFO("Time Cost: {}s.", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() * 0.001);
//
//	if (succ) {
//		_reg_flag_xm = true;
//		LOG_DEBUG("Calculate Online Car Pose Success!");
//		return true;
//	}
//	else {
//		_reg_flag_xm = false;
//		LOG_ERROR("Calculate Online Car Pose Failed!");
//		return false;
//	}
//}

Eigen::Matrix4d CarLocSystem3D::getCarInspectPose() {

	if (_reg_flag_xm) 
	{
		Eigen::Matrix4d m_pose_cad_cam0 = _pose_template_base2 * _pose_bc;
		Eigen::Matrix4d pose_cad_c1 = m_pose_cad_cam0 * _pose_cam0_cam1;

		//@xlqu修改：求得的结果应为cad1<-cad0
		Eigen::Matrix4d pose_cad0_cad1 = m_pose_cad_cam0 * pose_cad_c1.inverse();
		//光追返回结果
		//Eigen::Matrix4d pose_cad1_base2 = _pose_cam0_cam1.inverse() * _pose_template_base2;


		std::stringstream ss;
		ss << pose_cad0_cad1;
		LOG_DEBUG("Return Car Inspect Pose:\n{}", ss.str());

		return pose_cad0_cad1;
	}
	else {
		LOG_ERROR("getCarInspectPose: pose_cad0_cad1 Failed, Return Eigen::Matrix4d::Zero() Instead!");
		return Eigen::Matrix4d::Zero();
	}
}
std::string CarLocSystem3D::getSideString(const SIDE_ENUM& car_side) {
	static const std::unordered_map<SIDE_ENUM, std::string> SIDE_STRINGS = {
		{SIDE_ENUM::Unknown, "Unknown"},
		{SIDE_ENUM::Left, "Left"},
		{SIDE_ENUM::Right, "Right"}
	};

	auto it = SIDE_STRINGS.find(car_side);
	if (it != SIDE_STRINGS.end()) {
		return it->second;
	}
	return "Unknown";
}

SIDE_ENUM CarLocSystem3D::getSideEnum(const std::string& car_side_str) {
	static const std::unordered_map<std::string, SIDE_ENUM> SIDE_ENUNS = {
		{"Unknown", SIDE_ENUM::Unknown},
		{"Left", SIDE_ENUM::Left},
		{"Right", SIDE_ENUM::Right}
	};

	auto it = SIDE_ENUNS.find(car_side_str);
	if (it != SIDE_ENUNS.end()) {
		return it->second;
	}
	return SIDE_ENUM::Unknown;
}

std::string CarLocSystem3D::getPartString(const PART_ENUM& part_enum) {
	static const std::unordered_map<PART_ENUM, std::string> PART_STRINGS = {
		{PART_ENUM::Unknown, "Unknown"},
        {PART_ENUM::FrontLongitudinalBeam, "FrontLongitudinalBeam"},
        {PART_ENUM::RearWheelHouse, "RearWheelHouse"}
	};

	auto it = PART_STRINGS.find(part_enum);
	if (it != PART_STRINGS.end()) {
		return it->second;
	}
	return "Unknown";
}

PART_ENUM CarLocSystem3D::getPartEnum(const std::string& part_str) {
	static const std::unordered_map<std::string, PART_ENUM> PART_ENUMS = {
		{"Unknown", PART_ENUM::Unknown},
        {"FrontLongitudinalBeam", PART_ENUM::FrontLongitudinalBeam},
        {"RearWheelHouse", PART_ENUM::RearWheelHouse}
	};

	auto it = PART_ENUMS.find(part_str);
	if (it != PART_ENUMS.end()) {
		return it->second;
	}
	return PART_ENUM::Unknown;
}
