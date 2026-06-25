README


Instructions for how to run the FFT dune wavelength pipeline 

Pre-processing: 
Make polygons bounding all regions of interest and put them in a folder with each KMZ named for the region 
Run the make_sampling_points_from_kmz.py script 

Run the pipeline: 
Example usage: 

python run_pipeline_from_points.py   --points_csv sampling_points_0p25deg_all_regions.csv   --out_csv results/modes_multimode.csv   --tiles_dir tiles   --diag_dir outputs

Final CSV goes in "result"
Images go in "tiles"
Figures go in "outputs" 

Need to post process the modes, but all the relevant information is in the CSV 


