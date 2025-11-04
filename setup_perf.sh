#!/bin/bash
# Setup script for perf profiling
# Run this once: sudo ./setup_perf.sh

echo "Setting kernel.perf_event_paranoid to 1..."
echo "This allows perf to collect user-space performance data without root."
echo ""

sysctl -w kernel.perf_event_paranoid=1

echo ""
echo "Current setting:"
cat /proc/sys/kernel/perf_event_paranoid

echo ""
echo "To make this permanent, add to /etc/sysctl.conf:"
echo "kernel.perf_event_paranoid=1"
