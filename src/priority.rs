//! Scope-based thread priority management for long-running operations
//!
//! On Windows, sets thread priority to IDLE for the lifetime of the PriorityManager,
//! restoring to NORMAL on drop.
//!
//! On macOS, sets QoS (Quality of Service) class to UTILITY for long-running work,
//! restoring the original QoS on drop.
//!
//! This makes long-running operations less intrusive to user-facing tasks.

#[cfg(windows)]
mod platform_impl {
    use windows::Win32::System::Threading::{
        GetCurrentThread, GetThreadPriority, SetThreadPriority, THREAD_PRIORITY,
        THREAD_PRIORITY_IDLE,
    };

    pub struct PriorityManager {
        original_priority: THREAD_PRIORITY,
    }

    impl PriorityManager {
        pub fn new() -> Self {
            unsafe {
                let thread = GetCurrentThread();
                let original_priority = THREAD_PRIORITY(GetThreadPriority(thread));
                let _ = SetThreadPriority(thread, THREAD_PRIORITY_IDLE);
                Self { original_priority }
            }
        }
    }

    impl Drop for PriorityManager {
        fn drop(&mut self) {
            unsafe {
                let thread = GetCurrentThread();
                let _ = SetThreadPriority(thread, self.original_priority);
            }
        }
    }
}

#[cfg(target_os = "macos")]
mod platform_impl {
    use std::ffi::c_void;
    use std::os::raw::{c_int, c_uint};

    // Darwin QoS (Quality of Service) classes
    #[allow(dead_code)]
    const QOS_CLASS_USER_INTERACTIVE: c_uint = 0x21;
    #[allow(dead_code)]
    const QOS_CLASS_USER_INITIATED: c_uint = 0x19;
    #[allow(dead_code)]
    const QOS_CLASS_DEFAULT: c_uint = 0x15;
    const QOS_CLASS_UTILITY: c_uint = 0x11;
    #[allow(dead_code)]
    const QOS_CLASS_BACKGROUND: c_uint = 0x09;
    const QOS_CLASS_UNSPECIFIED: c_uint = 0x00;

    #[allow(non_camel_case_types)]
    type pthread_t = *mut c_void;

    unsafe extern "C" {
        fn pthread_self() -> pthread_t;
        fn pthread_get_qos_class_np(
            thread: pthread_t,
            qos_class: *mut c_uint,
            relative_priority: *mut c_int,
        ) -> c_int;
        fn pthread_set_qos_class_self_np(qos_class: c_uint, relative_priority: c_int) -> c_int;
    }

    pub struct PriorityManager {
        original_qos: c_uint,
        original_priority: c_int,
    }

    impl PriorityManager {
        pub fn new() -> Self {
            unsafe {
                let mut original_qos = QOS_CLASS_UNSPECIFIED;
                let mut original_priority = 0;

                // Get current QoS class
                let thread = pthread_self();
                let _ = pthread_get_qos_class_np(thread, &mut original_qos, &mut original_priority);

                // Set to UTILITY class for long-running work
                let _ = pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);

                Self {
                    original_qos,
                    original_priority,
                }
            }
        }
    }

    impl Drop for PriorityManager {
        fn drop(&mut self) {
            unsafe {
                // Restore original QoS class
                let _ = pthread_set_qos_class_self_np(self.original_qos, self.original_priority);
            }
        }
    }
}

#[cfg(not(any(windows, target_os = "macos")))]
mod platform_impl {
    pub struct PriorityManager;

    impl PriorityManager {
        #[inline]
        pub fn new() -> Self {
            Self
        }
    }
}

pub use platform_impl::PriorityManager;
