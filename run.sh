python_src='src/main.py'

image_ext='JPG'
fps=30
resize_fct=4
logo='data/logo_polimi.jpg'
codec='avc1'

image_dir='data/p1'
output_name='results/belvedere_time_lapse_p1'
python $python_src \
    -i $image_dir \
    -o $output_name \
    --image_ext $image_ext \
    --fps $fps \
    --resize_fct $resize_fct \
    --logo $logo


image_dir='data/p2'
output_name='results/belvedere_time_lapse_p2'
python $python_src \
    -i $image_dir \
    -o $output_name \
    --image_ext $image_ext \
    --fps $fps \
    --resize_fct $resize_fct \
    --logo $logo    