package engine

// TODO: Introductory line what this code does and reference to paper where this was introduced.
//   - see minimizer.go for reference

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// Some generic parameter values which can be overwritten bei init()
// TODO: explain what they are.
// TODO: Check if they are actually good.
var (
	massVPO             =   1.0	// TODO: massVPO of the ??? What units? What is reasonable here?
	stepsizeVPO 		= 0.01 // dt effective time step
	//MaxForce          = 100.0	// TODO: maximal allowed force???
	DmSamplesVPO int    =  10   	// number of dm to keep for convergence check
	StopMaxDmVPO float64 =  1e-6 	// stop minimizer if sampled dm is smaller than this
)

// Initialization of the function. Can override variables.
func init() { 
	DeclFunc("VPOMinimize", VPOMinimize, "Use gradient descent method to zero the forces (or energy gradients)")
	DeclVar("massVPO", &massVPO, "Mass used in the VPO minimizer")
	DeclVar("stepsizeVPO", &stepsizeVPO, "stepsizeVPO used in the VPO minimizer")
	//DeclVar("MaxForce", &MaxForce, "MaxForce")
	DeclVar("VPOMinimizerStop", &StopMaxDmVPO, "Stopping max dM for VPOMinimize")
	DeclVar("VPOMinimizerSamples", &DmSamplesVPO, "Number of max dM to collect for VPOMinimize convergence check.")
}

// fixed length FIFO. Items can be added but not removed. Copied from minimizer.
type fifoRingVPO struct {
	count int
	tail  int // index to put next item. Will loop to 0 after exceeding length
	data  []float64
}

func FifoRingVPO(length int) fifoRingVPO {
	return fifoRingVPO{data: make([]float64, length)}
}

func (r *fifoRingVPO) Add(item float64) {
	r.data[r.tail] = item
	r.count++
	r.tail = (r.tail + 1) % len(r.data)
	if r.count > len(r.data) {
		r.count = len(r.data)
	}
}

func (r *fifoRingVPO) Max() float64 {
	max := r.data[0]
	for i := 1; i < r.count; i++ {
		if r.data[i] > max {
			max = r.data[i]
		}
	}
	return max
}

// objects that need to be stored for next iteration step
type VPOMinimizer struct {
	f *data.Slice // force
	v *data.Slice // velocity
	lastDm fifoRingVPO
}

// VPOMinimizer step
func (mini *VPOMinimizer) Step() {

	// Magnetization m(t)
	m := M.Buffer()
	size := m.Size() // returns 3-vector with number of cells in every direction as entries


	// Force F(t), recycled from previous step (because it is a byproduct of the iteration process) 
	//  - If this is the first step (nil), compute directly from m.
	if mini.f == nil {	// make sure this is not empty upon first usage
		mini.f = cuda.Buffer(3, size)
		SetEffectiveField(mini.f)  // force is given by the field B_eff
		cuda.TangentSpaceProjection(mini.f, m) // project force onto tangent space of m
	}
	f := mini.f // this is a convenient shortcut, nothing more
	// TODO: Can we pull this shortcut up first or does mess up the pointer nature somehow? How does Go react?


	// Modified velocity vtilde(t-dt) = v(t-dt) + (dt/2m)F(t-dt), recycled from previous step (because it is a byproduct of the iteration process) 
	//  - If this is the first step (nil), set to zero.
	if mini.v == nil {
		mini.v = cuda.Buffer(3, size)
		cuda.Zero3(mini.v) // in cuda/slice.go > Zero(s *data.Slice)
	}
	v := mini.v // this is a convenient shortcut, nothing more

	
	// Compute velocity v(t) = vtilde(t-dt) + (dt/2m)F(t)
	cuda.Madd2(v, v, f, 1., float32(0.5*stepsizeVPO/massVPO)) // --> cuda/madd2.cu --> dst[i] = fac1*src1[i] + fac2*src2[i];
	// QUESTION: Where is stepsizeVPO defined? Which value? How to change? Why not dynamic?


	// Compute vtilde(t)
	//  - Project v(t) on direction parallel to F(t)
	//  - and add the extra term (dt/2m)F(t)
	vf := cuda.Dot(v,f) / cuda.Dot(f,f) // vf = <v,f> / <f,f>
	if vf <= 0.0 {
		vf = 0.0
	}
	cuda.Madd2(v, f, f, vf, float32(0.5*stepsizeVPO/massVPO))
	// QUESTION: Is there no simple Scale function for this?


	// For convergence check: Save copy of the magnetization
	m0 := cuda.Buffer(3, size) 	// allocate memory
	defer cuda.Recycle(m0)	   	// purge once this function ends
	data.Copy(m0, m)			// copy the current magnetization to m0


	// Update magnetization
	// m(t+dt) = m(t) + vtilde(t) dt
	cuda.RotationRodrigues(m, v, float32(stepsizeVPO)) // --> cuda/RotationRodrigues.go


	// Rotate the velocity vtilde(t) so that it now is in tangent space of m(t+dt)
	cuda.TangentSpaceRotation(v, v, m, m0) 
	// QUESTION: This is just a rotation, nothing special about tangent space. Should we have a rotation function instead, which rotates from m0 to m?


	// Calculate the force f(t+dt) (= effective field) after the update and project onto tangent space
	SetEffectiveField(f)
	cuda.TangentSpaceProjection(f, m) // --> cuda/TangentSpaceProjection.go


	// Calculate the maximal change in m and add it to the fifoRing-list
	// Needed for convergence check 
	dm := m0 // this is just for readability
	cuda.Madd2(dm, m, m0, 1., -1.)
	max_dm := cuda.MaxVecNorm(dm)
	mini.lastDm.Add(max_dm)
	setLastErr(mini.lastDm.Max()) // report maxDm to user as LastErr

	// End of this iteration step
	NSteps++
}

// Free
func (mini *VPOMinimizer) Free() {
	mini.f.Free()
	mini.v.Free()
}

// The main part of this function: Based on engine/minimizer.go
func VPOMinimize() {
	Refer("todo")
	SanityCheck()
	// Save the settings we are changing...
	prevType := solvertype
	prevFixDt := FixDt
	prevPrecess := Precess
	t0 := Time

	relaxing = true // disable temperature noise

	// ...to restore them later. Read as "defer ..." = "when function ends, do ..."
	defer func() {
		SetSolver(prevType)
		FixDt = prevFixDt
		Precess = prevPrecess
		Time = t0

		relaxing = false
	}()

	// disable precession for torque calculation
	Precess = false 

	// remove previous stepper
	if stepper != nil {
		stepper.Free()
	}

	// set stepper to the VPOMinimizer
	mini := VPOMinimizer{
		f: nil,
		v: nil,
		lastDm: FifoRingVPO(DmSamplesVPO)}
	stepper = &mini

	// break condition: change of magnetization is below a reasonable threshold
	cond := func() bool {
		return (mini.lastDm.count < DmSamplesVPO || mini.lastDm.Max() > StopMaxDmVPO)
	}

	RunWhile(cond)
	pause = true
}
