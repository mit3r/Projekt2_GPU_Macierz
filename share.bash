# Compress ./profiler_output to ./profiler_output.tar.gz
tar -czvf profiler_output.tar.gz profiler_output

# Move the compressed file to ~/public_html/
mv profiler_output.tar.gz ~/public_html/