# run_synth.tcl  —  Vitis HLS synthesis project for dbn_inout (quantised DBN kernel)
#
# Run from the hls_dbn/ directory:
#   vitis_hls run_synth.tcl
#   vitis_hls run_synth.tcl -tclargs csim     # csim only
#   vitis_hls run_synth.tcl -tclargs synth    # csim + synthesis
#   vitis_hls run_synth.tcl -tclargs cosim    # csim + synthesis + cosim

set PROJECT_NAME  "dbn_synth"
set SOLUTION_NAME "solution1"
set PART          "xc7z020clg484-1"
set CLOCK_NS      "10"

if { $argc > 0 } {
    set MODE [lindex $argv 0]
} else {
    set MODE "csim"
}

open_project  $PROJECT_NAME
set_top       dbn_inout
open_solution $SOLUTION_NAME -flow_target vivado
set_part      $PART
create_clock  -period $CLOCK_NS -name default

# Kernel source
add_files dbn_inout.cpp -cflags "-std=c++14"
add_files dbn_inout.hpp -cflags "-std=c++14"
add_files params.hpp    -cflags "-std=c++14"

# Testbench (reuses dbn_tb.cpp for csim verification of the kernel)
add_files -tb dbn_tb.cpp -cflags "-std=c++14"

if { $MODE eq "csim" || $MODE eq "synth" || $MODE eq "cosim" } {
    csim_design -argv "../../../../fixed_export/abu-airport-1"
}
if { $MODE eq "synth" || $MODE eq "cosim" } {
    csynth_design
}
if { $MODE eq "cosim" } {
    cosim_design -argv "../../../../fixed_export/abu-airport-1"
}

close_project