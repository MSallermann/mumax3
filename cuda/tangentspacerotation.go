package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func TangentSpaceRotation(v, v0, m, m0 *data.Slice) {

	util.Argument(v.NComp() == 3 && v0.NComp() == 3 && m.NComp() == 3 && m0.NComp() == 3)
	util.Argument(v.Len() == v0.Len() && v.Len() == m.Len() && v.Len() == m0.Len())

	N := v.Len()
	cfg := make1DConf(N)

	k_tangentspacerotation_async(
		 v.DevPtr(X),  v.DevPtr(Y),  v.DevPtr(Z),
		v0.DevPtr(X), v0.DevPtr(Y), v0.DevPtr(Z),
		 m.DevPtr(X),  m.DevPtr(Y),  m.DevPtr(Z),
		m0.DevPtr(X), m0.DevPtr(Y), m0.DevPtr(Z),
		N, cfg)
}
