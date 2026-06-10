# Compress ./profiler_output to ./profiler_output.tar.gz
tar -czvf profiler_outputs.tar.gz profiler_outputs

# Move the compressed file to ~/public_html/
mv profiler_outputs.tar.gz ~/public_html/ 