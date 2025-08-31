# Matrix Multiplication Scaling Analysis
# Usage: gnuplot scaling.plt
# Input: scaling_data.txt (CSV format: size,algorithm,time_ns,gflops)

set terminal png size 1400,700 font "Arial,12"
set output 'scaling_analysis.png'

set multiplot layout 1,2 title "Matrix Multiplication Scaling Analysis" font "Arial,16"

# Set up data parsing
set datafile separator ","

# Plot 1: Execution Time vs Matrix Size (log-log scale)
set logscale xy
set xlabel "Matrix Size (N×N)" font "Arial,12"
set ylabel "Execution Time (nanoseconds)" font "Arial,12"
set title "Execution Time Scaling\n(shows cache effects)" font "Arial,14"
set grid
set key left top

# Plot time data for each algorithm
plot '< grep "naive" scaling_data.txt' using 1:4 with linespoints \
     linewidth 2 pointsize 1.2 linecolor rgb "red" title "Naive (row×row)", \
     '< grep "optimized" scaling_data.txt' using 1:4 with linespoints \
     linewidth 2 pointsize 1.2 linecolor rgb "blue" title "Optimized (cache-friendly)"

# Plot 2: Performance vs Matrix Size (GFLOP/s)
unset logscale xy
set logscale x
set xlabel "Matrix Size (N×N)" font "Arial,12"
set ylabel "Performance (GFLOP/s)" font "Arial,12"
set title "Performance Scaling\n(higher is better)" font "Arial,14"
set grid
set key left bottom

# Plot GFLOP/s data for each algorithm
plot '< grep "naive" scaling_data.txt' using 1:5 with linespoints \
     linewidth 2 pointsize 1.2 linecolor rgb "red" title "Naive (row×row)", \
     '< grep "optimized" scaling_data.txt' using 1:5 with linespoints \
     linewidth 2 pointsize 1.2 linecolor rgb "blue" title "Optimized (cache-friendly)"

unset multiplot

# Create separate files for speedup calculation
system("grep 'naive' scaling_data.txt | cut -d, -f1,5 > naive.tmp")
system("grep 'optimized' scaling_data.txt | cut -d, -f1,5 > optimized.tmp")

# Create a second plot showing speedup
set terminal png size 800,600 font "Arial,12"
set output 'speedup_analysis.png'

set logscale x
unset logscale y
set xlabel "Matrix Size (N×N)" font "Arial,12"
set ylabel "Speedup Factor" font "Arial,12"
set title "Performance Improvement\n(Optimized vs Naive)" font "Arial,14"
set grid
set key right bottom

# Plot speedup using temporary files  
plot '< paste naive.tmp optimized.tmp' using 1:($3/$2) with linespoints \
     linewidth 3 pointsize 1.5 linecolor rgb "green" title "Speedup (optimized/naive)"

# Clean up temporary files
system("rm -f naive.tmp optimized.tmp")

print ""
print "Generated plots:"
print "  - scaling_analysis.png: Time and performance scaling"
print "  - speedup_analysis.png: Speedup analysis"
print ""
print "Expected observations:"
print "  - Performance drops at cache boundaries (32×32, 128×128, 512×512)"
print "  - Optimized version shows smaller performance drops"
print "  - Speedup increases with matrix size (better cache utilization)"