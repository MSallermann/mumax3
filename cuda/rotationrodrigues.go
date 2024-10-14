package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func RotationRodrigues(m, Beff *data.Slice, dt float32) {

	util.Argument(m.NComp() == 3 && Beff.NComp() == 3)
	util.Argument(m.Len() == Beff.Len())

	N := Beff.Len()
	cfg := make1DConf(N)

	k_rotationrodrigues_async(
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
		dt, N, cfg)
}
