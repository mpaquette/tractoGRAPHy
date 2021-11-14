# tractoGRAPHy  
Graph-based alternative to probabilistic tractography for diffusion MRI connectivity matrix.  


**Connectivity matrix from a fODF weighted graph: An alternative to probabilistic tractography**  
by  
Michael Paquette, Cornelius Eichner, and Alfred Anwander  
Submitted to ISMRM 2022  


### Naive graph approach:  
For each voxel inside a "tractography" mask (i.e. strict WM mask):  
- Project the fODF to a fine sphere  
- Assign each orientation of the sphere to one of 26 neighboor in a 3x3x3 cube  
- Sum and normalize fODF value for each neighbor to compute probability of connection  
- You can threshold the fODF based on a percentage of their maxima to nullify probability in direction perpendicular to maximas while retaining some of the lobes fanning  
Build a directed graph with a node for each voxel and these probability of connection  
Connectivity matrices are built from a label map using shortest paths  

This method is simple and fast but paths have no orientational constraint, they can take very sharp turns  
Also, having crossings penalise the probability of propagation even if the taken direction is very strong since the entire ODF sum to 1.  


### Oriented graph approach:  
For each voxel inside a "tractography" mask (i.e. strict WM mask):  
- Project the fODF to a fine sphere  
- Assign each orientation of the sphere to one of 26 neighboor in a 3x3x3 cube  
- Decide on a cone size to restrict orientation into  
- Centering a cone around each orientation from each neighboor, compute a mask over the neighboor point assignement  
- Sum and normalize fODF value for each neighbor for each cone orientation to compute the probability of connection for each orientation.  
Build a directed graph with a 26 node for each voxel (one for each neighboor) and these probability of connection  
Connectivity matrices are built from a label map using shortest paths  

Path propagation is now retricted inside a cone similarly to probabilistic tractography.  
The orientation of the cone depend on the angle of arrival into a given voxel (i.e. which neighboor it came from).  


```
compute_probability_naive_graph.py  
```
Compute and save the neighboor connection probability for naive graph construction.  
Needs a mask, an fODF field (in tournier07/Mrtrix3 format) and a relative ODF threshold.  

```
build_naive_graph.py  
```
Compute and save naive graph construction.  
Needs a mask and a probability map (from compute_probability_naive_graph.py).  

```
compute_shortest_path_naive_graph.py  
```
Compute shortest paths and connectivity matrices for naive graph.  
Needs a naive graph, a mask, a label map and a target_type.  

```
compute_probability_oriented_graph.py  
```
Compute and save the neighboor connection probability for oriented graph construction.  
Needs a mask, an fODF field (in tournier07/Mrtrix3 format), a relative ODF threshold and a cone half-angle.  

```
build_oriented_graph.py  
```
Compute and save oriented graph construction.  
Needs a mask, a probability map (from compute_probability_oriented_graph.py) and the cone half-angle that was used.  

```
compute_shortest_path_oriented_graph.py  
```
Compute shortest paths and connectivity matrices for oriented graph.  
Needs a oriented graph, a mask, a label map and a target_type.  

```
inspect_graph.py  
```
Give information on the graph type.   


Test data created from [Fiberfox replications of the FiberCup datasets](https://www.nitrc.org/frs/?group_id=627)  

test_data/results included generated probability maps, graphs, connectomes and shortest paths for naive, oriented, for COM and for ROI.  

