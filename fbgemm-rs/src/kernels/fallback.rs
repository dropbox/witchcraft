use super::{ref_kernel, GemmParams, KernelFn};

unsafe fn kernel_1(gp: *mut GemmParams) {
    ref_kernel(1, gp)
}
unsafe fn kernel_2(gp: *mut GemmParams) {
    ref_kernel(2, gp)
}
unsafe fn kernel_3(gp: *mut GemmParams) {
    ref_kernel(3, gp)
}
unsafe fn kernel_4(gp: *mut GemmParams) {
    ref_kernel(4, gp)
}
unsafe fn kernel_5(gp: *mut GemmParams) {
    ref_kernel(5, gp)
}
unsafe fn kernel_6(gp: *mut GemmParams) {
    ref_kernel(6, gp)
}

pub static KERNELS: [Option<KernelFn>; 15] = [
    None,           // 0 (unused)
    Some(kernel_1), // 1
    Some(kernel_2), // 2
    Some(kernel_3), // 3
    Some(kernel_4), // 4
    Some(kernel_5), // 5
    Some(kernel_6), // 6
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
];
