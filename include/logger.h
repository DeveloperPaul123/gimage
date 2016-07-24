#ifndef _LOGGER_H
#define _LOGGER_H

#include <ostream>
#include <memory> 
#include <string>

//Define a file name macro so the whole path to the file isn't printed in the log.
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

/**
* Logging class. Used in the log macros to save logging data
* to a file or print it to a stream. 
*/
class GIMAGE_EXPORT logger_t
{
public:
	static bool is_activated;
	static std::auto_ptr < std::ostream >
		outstream_helper_ptr;
	static std::ostream * outstream;
	logger_t(); 
private:
	logger_t(const logger_t &);
	logger_t & operator= (const logger_t &);
};

extern logger_t & logger();

#define LOG(name) do {\
	if (logger().is_activated ){\
		*logger().outstream << __FILENAME__ \
		<< " [" << __LINE__ << "] : " << #name \
		<< " = " << (name) << std::endl;\
				}\
} while (false)

#define LOG_INFO(name) do {\
	if(logger().is_activated) {\
	*logger().outstream << __FILENAME__ \
	<< " [" << __LINE__ << "] : " << (name) << std::endl;\
		}\
} while(false)

#define LOG_VERBOSE(name) do {\
	if(logger().is_activated) {\
		*logger().outstream << (name) << std::endl;}\
} while (false)

namespace logger_n {
	template < typename T1, typename T2, \
		typename T3, typename T4 >
		void put_debug_info(logger_t & log, \
		T1 const & t1, T2 const & t2, \
		T3 const & t3, T4 const & t4)
	{
		if (log.is_activated)
		{
			*(log.outstream) << t1 << " (" \
				<< t2 << ") : ";
			*(log.outstream) << t3 << " = " \
				<< t4 << std::endl;
		}
	}
}

#define LOG_FN(name) logger_n::put_debug_info ( \
	logger(), __FILE__, __LINE__, #name, (name) )
// place for user defined logger formating data
#define LOG_ON() do { \
   logger().is_activated = true; } while(false)
#define LOG_OFF() do { \
   logger().is_activated = false; } while(false)
#if defined(CLEANLOG)
#undef LOG
#undef LOG_ON
#undef LOG_OFF
#undef LOG_FN
#define LOG(name) do{}while(false)
#define LOG_FN(name) do{}while(false)
#define LOG_ON() do{}while(false)
#define LOG_OFF() do{}while(false)
#endif

#endif