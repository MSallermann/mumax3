package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

var shiftbytes_code cu.Function

type shiftbytes_args struct {
	arg_dst unsafe.Pointer
	arg_src unsafe.Pointer
	arg_N0  int
	arg_N1  int
	arg_N2  int
	arg_sh0 int
	arg_sh1 int
	arg_sh2 int
	argptr  [8]unsafe.Pointer
}

// Wrapper for shiftbytes CUDA kernel, asynchronous.
func k_shiftbytes_async(dst unsafe.Pointer, src unsafe.Pointer, N0 int, N1 int, N2 int, sh0 int, sh1 int, sh2 int, cfg *config, str cu.Stream) {
	if shiftbytes_code == 0 {
		shiftbytes_code = fatbinLoad(shiftbytes_map, "shiftbytes")
	}

	var _a_ shiftbytes_args

	_a_.arg_dst = dst
	_a_.argptr[0] = unsafe.Pointer(&_a_.arg_dst)
	_a_.arg_src = src
	_a_.argptr[1] = unsafe.Pointer(&_a_.arg_src)
	_a_.arg_N0 = N0
	_a_.argptr[2] = unsafe.Pointer(&_a_.arg_N0)
	_a_.arg_N1 = N1
	_a_.argptr[3] = unsafe.Pointer(&_a_.arg_N1)
	_a_.arg_N2 = N2
	_a_.argptr[4] = unsafe.Pointer(&_a_.arg_N2)
	_a_.arg_sh0 = sh0
	_a_.argptr[5] = unsafe.Pointer(&_a_.arg_sh0)
	_a_.arg_sh1 = sh1
	_a_.argptr[6] = unsafe.Pointer(&_a_.arg_sh1)
	_a_.arg_sh2 = sh2
	_a_.argptr[7] = unsafe.Pointer(&_a_.arg_sh2)

	args := _a_.argptr[:]
	cu.LaunchKernel(shiftbytes_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, str, args)
}

// Wrapper for shiftbytes CUDA kernel, synchronized.
func k_shiftbytes(dst unsafe.Pointer, src unsafe.Pointer, N0 int, N1 int, N2 int, sh0 int, sh1 int, sh2 int, cfg *config) {
	str := stream()
	k_shiftbytes_async(dst, src, N0, N1, N2, sh0, sh1, sh2, cfg, str)
	syncAndRecycle(str)
}

var shiftbytes_map = map[int]string{0: "",
	20: shiftbytes_ptx_20,
	30: shiftbytes_ptx_30,
	35: shiftbytes_ptx_35}

const (
	shiftbytes_ptx_20 = `
.version 3.1
.target sm_20
.address_size 64


.visible .entry shiftbytes(
	.param .u64 shiftbytes_param_0,
	.param .u64 shiftbytes_param_1,
	.param .u32 shiftbytes_param_2,
	.param .u32 shiftbytes_param_3,
	.param .u32 shiftbytes_param_4,
	.param .u32 shiftbytes_param_5,
	.param .u32 shiftbytes_param_6,
	.param .u32 shiftbytes_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .s16 	%rc<2>;
	.reg .s32 	%r<38>;
	.reg .s64 	%rd<9>;


	ld.param.u64 	%rd3, [shiftbytes_param_0];
	ld.param.u64 	%rd4, [shiftbytes_param_1];
	ld.param.u32 	%r4, [shiftbytes_param_2];
	ld.param.u32 	%r5, [shiftbytes_param_3];
	ld.param.u32 	%r6, [shiftbytes_param_4];
	ld.param.u32 	%r7, [shiftbytes_param_5];
	ld.param.u32 	%r8, [shiftbytes_param_6];
	ld.param.u32 	%r9, [shiftbytes_param_7];
	cvta.to.global.u64 	%rd1, %rd3;
	cvta.to.global.u64 	%rd2, %rd4;
	.loc 2 10 1
	mov.u32 	%r10, %ntid.z;
	mov.u32 	%r11, %ctaid.z;
	mov.u32 	%r12, %tid.z;
	mad.lo.s32 	%r1, %r10, %r11, %r12;
	.loc 2 11 1
	mov.u32 	%r13, %ntid.y;
	mov.u32 	%r14, %ctaid.y;
	mov.u32 	%r15, %tid.y;
	mad.lo.s32 	%r2, %r13, %r14, %r15;
	.loc 2 12 1
	mov.u32 	%r16, %ntid.x;
	mov.u32 	%r17, %ctaid.x;
	mov.u32 	%r18, %tid.x;
	mad.lo.s32 	%r3, %r16, %r17, %r18;
	.loc 2 14 1
	setp.ge.s32 	%p1, %r2, %r5;
	setp.ge.s32 	%p2, %r1, %r4;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_2;

	.loc 2 18 1
	sub.s32 	%r19, %r1, %r7;
	mov.u32 	%r20, 0;
	.loc 3 238 5
	max.s32 	%r21, %r19, %r20;
	.loc 2 18 1
	add.s32 	%r22, %r4, -1;
	.loc 3 210 5
	min.s32 	%r23, %r21, %r22;
	.loc 2 18 1
	sub.s32 	%r24, %r2, %r8;
	.loc 3 238 5
	max.s32 	%r25, %r24, %r20;
	.loc 2 18 1
	add.s32 	%r26, %r5, -1;
	.loc 3 210 5
	min.s32 	%r27, %r25, %r26;
	.loc 2 18 1
	mad.lo.s32 	%r28, %r23, %r5, %r27;
	sub.s32 	%r29, %r3, %r9;
	.loc 3 238 5
	max.s32 	%r30, %r29, %r20;
	.loc 2 18 1
	add.s32 	%r31, %r6, -1;
	.loc 3 210 5
	min.s32 	%r32, %r30, %r31;
	.loc 2 18 1
	mad.lo.s32 	%r33, %r28, %r6, %r32;
	cvt.s64.s32 	%rd5, %r33;
	add.s64 	%rd6, %rd2, %rd5;
	mad.lo.s32 	%r34, %r1, %r5, %r2;
	mad.lo.s32 	%r35, %r34, %r6, %r3;
	cvt.s64.s32 	%rd7, %r35;
	add.s64 	%rd8, %rd1, %rd7;
	ld.global.u8 	%rc1, [%rd6];
	st.global.u8 	[%rd8], %rc1;

BB0_2:
	.loc 2 19 2
	ret;
}


`
	shiftbytes_ptx_30 = `
.version 3.1
.target sm_30
.address_size 64


.visible .entry shiftbytes(
	.param .u64 shiftbytes_param_0,
	.param .u64 shiftbytes_param_1,
	.param .u32 shiftbytes_param_2,
	.param .u32 shiftbytes_param_3,
	.param .u32 shiftbytes_param_4,
	.param .u32 shiftbytes_param_5,
	.param .u32 shiftbytes_param_6,
	.param .u32 shiftbytes_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .s16 	%rc<2>;
	.reg .s32 	%r<38>;
	.reg .s64 	%rd<9>;


	ld.param.u64 	%rd3, [shiftbytes_param_0];
	ld.param.u64 	%rd4, [shiftbytes_param_1];
	ld.param.u32 	%r4, [shiftbytes_param_2];
	ld.param.u32 	%r5, [shiftbytes_param_3];
	ld.param.u32 	%r6, [shiftbytes_param_4];
	ld.param.u32 	%r7, [shiftbytes_param_5];
	ld.param.u32 	%r8, [shiftbytes_param_6];
	ld.param.u32 	%r9, [shiftbytes_param_7];
	cvta.to.global.u64 	%rd1, %rd3;
	cvta.to.global.u64 	%rd2, %rd4;
	.loc 2 10 1
	mov.u32 	%r10, %ntid.z;
	mov.u32 	%r11, %ctaid.z;
	mov.u32 	%r12, %tid.z;
	mad.lo.s32 	%r1, %r10, %r11, %r12;
	.loc 2 11 1
	mov.u32 	%r13, %ntid.y;
	mov.u32 	%r14, %ctaid.y;
	mov.u32 	%r15, %tid.y;
	mad.lo.s32 	%r2, %r13, %r14, %r15;
	.loc 2 12 1
	mov.u32 	%r16, %ntid.x;
	mov.u32 	%r17, %ctaid.x;
	mov.u32 	%r18, %tid.x;
	mad.lo.s32 	%r3, %r16, %r17, %r18;
	.loc 2 14 1
	setp.ge.s32 	%p1, %r2, %r5;
	setp.ge.s32 	%p2, %r1, %r4;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_2;

	.loc 2 18 1
	sub.s32 	%r19, %r1, %r7;
	mov.u32 	%r20, 0;
	.loc 3 238 5
	max.s32 	%r21, %r19, %r20;
	.loc 2 18 1
	add.s32 	%r22, %r4, -1;
	.loc 3 210 5
	min.s32 	%r23, %r21, %r22;
	.loc 2 18 1
	sub.s32 	%r24, %r2, %r8;
	.loc 3 238 5
	max.s32 	%r25, %r24, %r20;
	.loc 2 18 1
	add.s32 	%r26, %r5, -1;
	.loc 3 210 5
	min.s32 	%r27, %r25, %r26;
	.loc 2 18 1
	mad.lo.s32 	%r28, %r23, %r5, %r27;
	sub.s32 	%r29, %r3, %r9;
	.loc 3 238 5
	max.s32 	%r30, %r29, %r20;
	.loc 2 18 1
	add.s32 	%r31, %r6, -1;
	.loc 3 210 5
	min.s32 	%r32, %r30, %r31;
	.loc 2 18 1
	mad.lo.s32 	%r33, %r28, %r6, %r32;
	cvt.s64.s32 	%rd5, %r33;
	add.s64 	%rd6, %rd2, %rd5;
	mad.lo.s32 	%r34, %r1, %r5, %r2;
	mad.lo.s32 	%r35, %r34, %r6, %r3;
	cvt.s64.s32 	%rd7, %r35;
	add.s64 	%rd8, %rd1, %rd7;
	ld.global.u8 	%rc1, [%rd6];
	st.global.u8 	[%rd8], %rc1;

BB0_2:
	.loc 2 19 2
	ret;
}


`
	shiftbytes_ptx_35 = `
.version 3.1
.target sm_35
.address_size 64


.visible .entry shiftbytes(
	.param .u64 shiftbytes_param_0,
	.param .u64 shiftbytes_param_1,
	.param .u32 shiftbytes_param_2,
	.param .u32 shiftbytes_param_3,
	.param .u32 shiftbytes_param_4,
	.param .u32 shiftbytes_param_5,
	.param .u32 shiftbytes_param_6,
	.param .u32 shiftbytes_param_7
)
{
	.reg .pred 	%p<6>;
	.reg .s16 	%rc<2>;
	.reg .s32 	%r<38>;
	.reg .s64 	%rd<9>;


	ld.param.u64 	%rd3, [shiftbytes_param_0];
	ld.param.u64 	%rd4, [shiftbytes_param_1];
	ld.param.u32 	%r4, [shiftbytes_param_2];
	ld.param.u32 	%r5, [shiftbytes_param_3];
	ld.param.u32 	%r6, [shiftbytes_param_4];
	ld.param.u32 	%r7, [shiftbytes_param_5];
	ld.param.u32 	%r8, [shiftbytes_param_6];
	ld.param.u32 	%r9, [shiftbytes_param_7];
	cvta.to.global.u64 	%rd1, %rd3;
	cvta.to.global.u64 	%rd2, %rd4;
	.loc 2 10 1
	mov.u32 	%r10, %ntid.z;
	mov.u32 	%r11, %ctaid.z;
	mov.u32 	%r12, %tid.z;
	mad.lo.s32 	%r1, %r10, %r11, %r12;
	.loc 2 11 1
	mov.u32 	%r13, %ntid.y;
	mov.u32 	%r14, %ctaid.y;
	mov.u32 	%r15, %tid.y;
	mad.lo.s32 	%r2, %r13, %r14, %r15;
	.loc 2 12 1
	mov.u32 	%r16, %ntid.x;
	mov.u32 	%r17, %ctaid.x;
	mov.u32 	%r18, %tid.x;
	mad.lo.s32 	%r3, %r16, %r17, %r18;
	.loc 2 14 1
	setp.ge.s32 	%p1, %r2, %r5;
	setp.ge.s32 	%p2, %r1, %r4;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_2;

	.loc 2 18 1
	sub.s32 	%r19, %r1, %r7;
	mov.u32 	%r20, 0;
	.loc 3 238 5
	max.s32 	%r21, %r19, %r20;
	.loc 2 18 1
	add.s32 	%r22, %r4, -1;
	.loc 3 210 5
	min.s32 	%r23, %r21, %r22;
	.loc 2 18 1
	sub.s32 	%r24, %r2, %r8;
	.loc 3 238 5
	max.s32 	%r25, %r24, %r20;
	.loc 2 18 1
	add.s32 	%r26, %r5, -1;
	.loc 3 210 5
	min.s32 	%r27, %r25, %r26;
	.loc 2 18 1
	mad.lo.s32 	%r28, %r23, %r5, %r27;
	sub.s32 	%r29, %r3, %r9;
	.loc 3 238 5
	max.s32 	%r30, %r29, %r20;
	.loc 2 18 1
	add.s32 	%r31, %r6, -1;
	.loc 3 210 5
	min.s32 	%r32, %r30, %r31;
	.loc 2 18 1
	mad.lo.s32 	%r33, %r28, %r6, %r32;
	cvt.s64.s32 	%rd5, %r33;
	add.s64 	%rd6, %rd2, %rd5;
	mad.lo.s32 	%r34, %r1, %r5, %r2;
	mad.lo.s32 	%r35, %r34, %r6, %r3;
	cvt.s64.s32 	%rd7, %r35;
	add.s64 	%rd8, %rd1, %rd7;
	ld.global.u8 	%rc1, [%rd6];
	st.global.u8 	[%rd8], %rc1;

BB0_2:
	.loc 2 19 2
	ret;
}


`
)
