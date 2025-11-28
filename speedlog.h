#pragma once
#include <iostream>
#include <string>
#include <memory>
#include <time.h>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <ctime>
#include <sstream>

#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/stdout_color_sinks.h" // or "../stdout_sinks.h" if no color needed
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"




class SpeedLogger
{
public:
	struct LoggerConfig
	{
		LoggerConfig(std::string path, std::string file_prefix) { logpath = path; logger_name_prefix = file_prefix; }
		std::string logpath = "./Log/";
		std::string logger_name_prefix = "mylog";
		// decide print to console or log file
		bool console = false;
		// decide the log level
		std::string level = "debug";
	};
	//单例模式
	static SpeedLogger* getInstance()
	{
		static SpeedLogger speedlogger;
		return &speedlogger;
	}

	std::shared_ptr<spdlog::logger> getLogger()
	{
		return m_logger;
	}

	bool init(const LoggerConfig& _config)
	{
		if (m_is_inited) return true;
		try {
			std::string logPath = _config.logpath;
			if (logPath.empty()) {
				logPath = "./Log/";
			}
			if (!std::filesystem::exists(logPath))
			{
				std::filesystem::create_directory(logPath);
			}
			//get the current time
			auto now = std::chrono::system_clock::now();
			std::time_t time = std::chrono::system_clock::to_time_t(now);

			//   ʽ  ʱ  Ϊ ַ   
			std::stringstream ss;
			struct tm currentTime;
			localtime_s(&currentTime, &time);
			ss << std::put_time(&currentTime, "%Y%m%d%H%M%S");

			// 获取当前线程ID
			std::stringstream thread_ss;
			thread_ss << std::this_thread::get_id();
			std::string thread_id_str = thread_ss.str();

			const std::string logger_name = _config.logger_name_prefix + "_" + ss.str() + "_" + thread_id_str;

			if (_config.console)
				m_logger = spdlog::stdout_color_st(logger_name); // single thread console output faster
			else
				//m_logger = spdlog::create_async<spdlog::sinks::basic_file_sink_mt>(logger_name, log_dir + "/" + logger_name + ".log"); // only one log file
				m_logger = spdlog::create_async<spdlog::sinks::rotating_file_sink_mt>(logger_name, logPath + "/" + logger_name + ".log", 5 * 1024 * 1024, 100); // multi part log files, with every part 5M, max 100 files

			// custom format
			m_logger->set_pattern("%Y-%m-%d %H:%M:%S.%f <thread %t> [%l] %s(%#): %v"); // with timestamp, thread_id, filename and line number

			auto log_level = spdlog::level::from_str(_config.level);
			m_logger->set_level(log_level);
			m_logger->flush_on(spdlog::level::info);
			spdlog::flush_every(std::chrono::seconds(1));
		}
		catch (const spdlog::spdlog_ex& ex)
		{
			std::cout << "Log initialization failed: " << ex.what() << std::endl;
			return false;
		}

		m_is_inited = true;
		return true;
	}

	void unInit()
	{
		spdlog::drop_all();
		spdlog::shutdown();

		m_is_inited = false;
	}

private:
	// make constructor private to avoid outside instance
	SpeedLogger() = default;

	~SpeedLogger()
	{
		if (m_is_inited) {
			unInit();
		}
	}

	SpeedLogger(const SpeedLogger&) = delete;
	SpeedLogger& operator=(const SpeedLogger&) = delete;

private:
	std::shared_ptr<spdlog::logger> m_logger;
	std::atomic_bool m_is_inited = false;
};

// use embedded macro to support file and line number
#define LOG_TRACE(...) SPDLOG_LOGGER_CALL(SpeedLogger::getInstance()->getLogger().get(), spdlog::level::trace, __VA_ARGS__)
#define LOG_DEBUG(...) SPDLOG_LOGGER_CALL(SpeedLogger::getInstance()->getLogger().get(), spdlog::level::debug, __VA_ARGS__)
#define LOG_INFO(...) SPDLOG_LOGGER_CALL(SpeedLogger::getInstance()->getLogger().get(), spdlog::level::info, __VA_ARGS__)
#define LOG_WARN(...) SPDLOG_LOGGER_CALL(SpeedLogger::getInstance()->getLogger().get(), spdlog::level::warn, __VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_LOGGER_CALL(SpeedLogger::getInstance()->getLogger().get(), spdlog::level::err, __VA_ARGS__)



