package engine

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/gui"
	"github.com/mumax/3/util"
	"log"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"time"
)

var (
	quants         = make(map[string]Slicer)                 // displayable
	renderQ        = "m"                                     // quantity to display
	params         = make(map[string]Param)                  // settable
	guiRegion      = -1                                      // currently addressed region
	usingX, usingY = 1, 2                                    // columns to plot
	KeepAlive      = func() time.Time { return time.Time{} } // overwritten by gui server
)

// displayable in GUI Parameters section
type Param interface {
	NComp() int
	Unit() string
	getRegion(int) []float64
	setRegion(int, []float64)
	IsUniform() bool
}

// data for html template
type guidata struct {
	Quants map[string]Slicer
	Params map[string]Param
}

// list of text box id's for component text boxes.
func (d *guidata) CompBoxIds(param string) []string {
	var e []string
	p := params[param]
	for i := 0; i < p.NComp(); i++ {
		e = append(e, fmt.Sprint("param_", param, i))
	}
	return e
}

// util for generating region numbers in GUI Parameters section.
func (d *guidata) MakeRange(min, max int) []int {
	l := make([]int, max-min)
	for i := range l {
		l[i] = min + i
	}
	return l
}

var gui_ *gui.Doc // use with caution, may not be inited yet.

// Start web gui on given port, blocks.
func Serve(port string) {
	data := &guidata{Quants: quants, Params: params}
	gui_ = gui.NewDoc(templText, data)
	gui := gui_
	KeepAlive = gui.KeepAlive

	http.Handle("/", gui)
	http.HandleFunc("/render/", serveRender)
	http.HandleFunc("/plot/", servePlot)

	// solver
	gui.OnEvent("break", inj(func() { pause = true }))
	gui.OnEvent("run", inj(func() { Run(gui.Value("runtime").(float64)) }))
	gui.OnEvent("steps", inj(func() { Steps(gui.Value("runsteps").(int)) }))
	gui.OnEvent("fixdt", inj(func() { Solver.FixDt = gui.Value("fixdt").(float64) }))
	gui.OnEvent("mindt", inj(func() { Solver.MinDt = gui.Value("mindt").(float64) }))
	gui.OnEvent("maxdt", inj(func() { Solver.MaxDt = gui.Value("maxdt").(float64) }))
	gui.OnEvent("maxerr", inj(func() { Solver.MaxErr = gui.Value("maxerr").(float64) }))
	gui.OnEvent("sel_render", func() { renderQ = gui.Value("sel_render").(string) })

	// display
	gui.SetValue("sel_render", renderQ)

	// gnuplot
	gui.OnEvent("usingX", func() { usingX = gui.Value("usingX").(int) })
	gui.OnEvent("usingY", func() { usingY = gui.Value("usingY").(int) })

	// setting parameters
	gui.SetValue("sel_region", guiRegion)
	gui.OnEvent("sel_region", func() { guiRegion = atoi(gui.Value("sel_region")) })

	for n, p := range params {
		n := n // closure caveats...
		p := p

		compIds := ((*guidata)(nil)).CompBoxIds(n)
		handler := func() {
			var cmd string
			if guiRegion == -1 {
				cmd = fmt.Sprintf("%v = (", n)
			} else {
				cmd = fmt.Sprintf("%v.setRegion(%v,", n, guiRegion)
			}
			if p.NComp() == 3 {
				cmd += fmt.Sprintf("vector(%v, %v, %v)",
					gui.Value(compIds[0]), gui.Value(compIds[1]), gui.Value(compIds[2]))
			} else {
				cmd += fmt.Sprint(gui.Value(compIds[0]))
			}
			cmd += ");"
			Inject <- func() { Eval(cmd, gui) }
		}
		for _, id := range compIds {
			gui.OnEvent(id, handler)
		}
	}

	// process
	gui.SetValue("gpu", fmt.Sprint(cuda.DevName, " (", (cuda.TotalMem)/(1024*1024), "MB)", ", CUDA ", cuda.Version))
	hostname, _ := os.Hostname()
	gui.SetValue("hostname", hostname)
	var memstats runtime.MemStats

	// periodically update time, steps, etc
	onrefresh := func() {

		// geometry
		size := globalmesh.Size()
		gui.SetValue("nx", size[2])
		gui.SetValue("ny", size[1])
		gui.SetValue("nz", size[0])
		cellSize := globalmesh.CellSize()
		gui.SetValue("cx", float32(cellSize[2]*1e9)) // in nm
		gui.SetValue("cy", float32(cellSize[1]*1e9))
		gui.SetValue("cz", float32(cellSize[0]*1e9))
		gui.SetValue("wx", float32(float64(size[2])*cellSize[2]*1e9))
		gui.SetValue("wy", float32(float64(size[1])*cellSize[1]*1e9))
		gui.SetValue("wz", float32(float64(size[0])*cellSize[0]*1e9))

		// solver
		gui.SetValue("time", fmt.Sprintf("%6e", Time))
		gui.SetValue("dt", fmt.Sprintf("%4e", Solver.Dt_si))
		gui.SetValue("step", Solver.NSteps)
		gui.SetValue("lasterr", fmt.Sprintf("%3e", Solver.LastErr))
		gui.SetValue("maxerr", Solver.MaxErr)
		gui.SetValue("mindt", Solver.MinDt)
		gui.SetValue("maxdt", Solver.MaxDt)
		gui.SetValue("fixdt", Solver.FixDt)
		if pause {
			gui.SetValue("solverstatus", "paused")
		} else {
			gui.SetValue("solverstatus", "running")
		}

		// display
		cachebreaker := "?" + fmt.Sprint(Solver.NSteps)
		gui.SetValue("render", "/render/"+renderQ+cachebreaker)

		// plot
		gui.SetValue("plot", "/plot/"+cachebreaker)

		// parameters
		for n, p := range params {
			if guiRegion == -1 {
				if p.IsUniform() {
					v := p.getRegion(0)
					for comp, id := range ((*guidata)(nil)).CompBoxIds(n) {
						gui.SetValue(id, fmt.Sprintf("%g", float32(v[comp])))
					}
				} else {
					for _, id := range ((*guidata)(nil)).CompBoxIds(n) {
						gui.SetValue(id, "")
					}
				}
			} else {
				v := p.getRegion(guiRegion)
				for comp, id := range ((*guidata)(nil)).CompBoxIds(n) {
					gui.SetValue(id, fmt.Sprintf("%g", float32(v[comp])))
				}
			}
		}

		// process
		gui.SetValue("walltime", fmt.Sprint(roundt(time.Since(StartTime))))
		runtime.ReadMemStats(&memstats)
		gui.SetValue("memstats", memstats.TotalAlloc/(1024))
	}

	gui.OnRefresh(func() { Inject <- onrefresh })

	log.Print(" =====\n open your browser and visit http://localhost", port, "\n =====\n")
	util.LogErr(http.ListenAndServe(port, nil))
	runtime.Gosched()
}

var StartTime = time.Now()

// round duration to 1s accuracy
func roundt(t time.Duration) time.Duration {
	return t - t%1e9
}

// returns a function that injects f into run loop
func inj(f func()) func() {
	return func() { Inject <- f }
}

func Eval(code string, gui *gui.Doc) {
	log.Println("eval", code)
	defer func() {
		err := recover()
		if err != nil {
			gui.SetValue("ErrorBox", fmt.Sprint(err))
			log.Println(err)
		}
	}()
	tree, err := World.Compile(code)
	if err == nil {
		tree.Eval()
	} else {
		gui.SetValue("paramErr", fmt.Sprint(err))
		log.Println(err)
	}
}

func atoi(x interface{}) int {
	i, err := strconv.Atoi(fmt.Sprint(x))
	util.LogErr(err)
	return i
}
