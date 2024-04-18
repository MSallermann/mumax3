package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"sync"
	"unsafe"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
)

// CUDA handle for copypath kernel
var copypath_code cu.Function

// Stores the arguments for copypath kernel invocation
type copypath_args_t struct {
	arg_Tx  unsafe.Pointer
	arg_Ty  unsafe.Pointer
	arg_Tz  unsafe.Pointer
	arg_mx  unsafe.Pointer
	arg_my  unsafe.Pointer
	arg_mz  unsafe.Pointer
	arg_Nx  int
	arg_Ny  int
	arg_Nz  int
	arg_noi int
	argptr  [10]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for copypath kernel invocation
var copypath_args copypath_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	copypath_args.argptr[0] = unsafe.Pointer(&copypath_args.arg_Tx)
	copypath_args.argptr[1] = unsafe.Pointer(&copypath_args.arg_Ty)
	copypath_args.argptr[2] = unsafe.Pointer(&copypath_args.arg_Tz)
	copypath_args.argptr[3] = unsafe.Pointer(&copypath_args.arg_mx)
	copypath_args.argptr[4] = unsafe.Pointer(&copypath_args.arg_my)
	copypath_args.argptr[5] = unsafe.Pointer(&copypath_args.arg_mz)
	copypath_args.argptr[6] = unsafe.Pointer(&copypath_args.arg_Nx)
	copypath_args.argptr[7] = unsafe.Pointer(&copypath_args.arg_Ny)
	copypath_args.argptr[8] = unsafe.Pointer(&copypath_args.arg_Nz)
	copypath_args.argptr[9] = unsafe.Pointer(&copypath_args.arg_noi)
}

// Wrapper for copypath CUDA kernel, asynchronous.
func k_copypath_async(Tx unsafe.Pointer, Ty unsafe.Pointer, Tz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, Nx int, Ny int, Nz int, noi int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("copypath")
	}

	copypath_args.Lock()
	defer copypath_args.Unlock()

	if copypath_code == 0 {
		copypath_code = fatbinLoad(copypath_map, "copypath")
	}

	copypath_args.arg_Tx = Tx
	copypath_args.arg_Ty = Ty
	copypath_args.arg_Tz = Tz
	copypath_args.arg_mx = mx
	copypath_args.arg_my = my
	copypath_args.arg_mz = mz
	copypath_args.arg_Nx = Nx
	copypath_args.arg_Ny = Ny
	copypath_args.arg_Nz = Nz
	copypath_args.arg_noi = noi

	args := copypath_args.argptr[:]
	cu.LaunchKernel(copypath_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("copypath")
	}
}

// maps compute capability on PTX code for copypath kernel.
var copypath_map = map[int]string{0: "",
	50: copypath_ptx_50}

// copypath PTX code for various compute capabilities.
const (
	copypath_ptx_50 = `
.version 7.5
.target sm_50
.address_size 64

	// .globl	copypath

.visible .entry copypath(
	.param .u64 copypath_param_0,
	.param .u64 copypath_param_1,
	.param .u64 copypath_param_2,
	.param .u64 copypath_param_3,
	.param .u64 copypath_param_4,
	.param .u64 copypath_param_5,
	.param .u32 copypath_param_6,
	.param .u32 copypath_param_7,
	.param .u32 copypath_param_8,
	.param .u32 copypath_param_9
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<23>;
	.reg .b64 	%rd<21>;


	ld.param.u64 	%rd4, [copypath_param_0];
	ld.param.u64 	%rd5, [copypath_param_1];
	ld.param.u64 	%rd6, [copypath_param_2];
	ld.param.u64 	%rd7, [copypath_param_3];
	ld.param.u64 	%rd8, [copypath_param_4];
	ld.param.u64 	%rd9, [copypath_param_5];
	ld.param.u32 	%r5, [copypath_param_6];
	ld.param.u32 	%r6, [copypath_param_7];
	ld.param.u32 	%r7, [copypath_param_8];
	ld.param.u32 	%r8, [copypath_param_9];
	mov.u32 	%r9, %ntid.x;
	mov.u32 	%r10, %ctaid.x;
	mov.u32 	%r11, %tid.x;
	mad.lo.s32 	%r1, %r10, %r9, %r11;
	mov.u32 	%r12, %ntid.y;
	mov.u32 	%r13, %ctaid.y;
	mov.u32 	%r14, %tid.y;
	mad.lo.s32 	%r2, %r13, %r12, %r14;
	mov.u32 	%r15, %ntid.z;
	mov.u32 	%r16, %ctaid.z;
	mov.u32 	%r17, %tid.z;
	mad.lo.s32 	%r3, %r16, %r15, %r17;
	setp.ge.s32 	%p1, %r1, %r5;
	setp.ge.s32 	%p2, %r2, %r6;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_4;

	cvta.to.global.u64 	%rd10, %rd4;
	mad.lo.s32 	%r18, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r18, %r5, %r1;
	div.s32 	%r19, %r7, %r8;
	div.s32 	%r20, %r3, %r19;
	setp.eq.s32 	%p6, %r20, 0;
	add.s32 	%r21, %r8, -1;
	setp.eq.s32 	%p7, %r20, %r21;
	or.pred  	%p8, %p6, %p7;
	mul.wide.s32 	%rd11, %r4, 4;
	add.s64 	%rd1, %rd10, %rd11;
	cvta.to.global.u64 	%rd12, %rd5;
	add.s64 	%rd2, %rd12, %rd11;
	cvta.to.global.u64 	%rd13, %rd6;
	add.s64 	%rd3, %rd13, %rd11;
	@%p8 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_2;

$L__BB0_3:
	mov.u32 	%r22, 0;
	st.global.u32 	[%rd1], %r22;
	st.global.u32 	[%rd2], %r22;
	st.global.u32 	[%rd3], %r22;
	bra.uni 	$L__BB0_4;

$L__BB0_2:
	cvta.to.global.u64 	%rd14, %rd7;
	add.s64 	%rd16, %rd14, %rd11;
	ld.global.nc.f32 	%f1, [%rd16];
	st.global.f32 	[%rd1], %f1;
	cvta.to.global.u64 	%rd17, %rd8;
	add.s64 	%rd18, %rd17, %rd11;
	ld.global.nc.f32 	%f2, [%rd18];
	st.global.f32 	[%rd2], %f2;
	cvta.to.global.u64 	%rd19, %rd9;
	add.s64 	%rd20, %rd19, %rd11;
	ld.global.nc.f32 	%f3, [%rd20];
	st.global.f32 	[%rd3], %f3;

$L__BB0_4:
	ret;

}

`
)
