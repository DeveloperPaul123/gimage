
#ifndef GIMAGE_EXPORT_H
#define GIMAGE_EXPORT_H

#ifdef GIMAGE_STATIC_DEFINE
#  define GIMAGE_EXPORT
#  define GIMAGE_NO_EXPORT
#else
#  ifndef GIMAGE_EXPORT
#    ifdef gimage_EXPORTS
        /* We are building this library */
#      define GIMAGE_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define GIMAGE_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef GIMAGE_NO_EXPORT
#    define GIMAGE_NO_EXPORT 
#  endif
#endif

#ifndef GIMAGE_DEPRECATED
#  define GIMAGE_DEPRECATED __declspec(deprecated)
#endif

#ifndef GIMAGE_DEPRECATED_EXPORT
#  define GIMAGE_DEPRECATED_EXPORT GIMAGE_EXPORT GIMAGE_DEPRECATED
#endif

#ifndef GIMAGE_DEPRECATED_NO_EXPORT
#  define GIMAGE_DEPRECATED_NO_EXPORT GIMAGE_NO_EXPORT GIMAGE_DEPRECATED
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define GIMAGE_NO_DEPRECATED
#endif

#endif
