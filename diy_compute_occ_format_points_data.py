import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import mesh_to_sdf
#os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ["OMP_NUM_THREADS"]="2"
import trimesh
import numpy as np
import glob
import multiprocessing as mp
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pyvirtualdisplay import Display

def sort_key(filename):
    return int(filename.split('_')[0])

def process_object(mesh_path, id, occ_save_dir, pcd_save_dir):
    
    occ_save_path=os.path.join(occ_save_dir,f"{id}.npz")
    
    if os.path.exists(occ_save_path):
       print("skipping %s" % (occ_save_path))
       return
    print(f'Start processing {id}.npz')

    tri_mesh=trimesh.load(mesh_path)  # Important!! Whether it is needed to multiply the factor 2
    tri_mesh.vertices = tri_mesh.vertices * 2.0
    vert = np.asarray(tri_mesh.vertices)
    
    tri_mesh.vertices=vert[:,:].copy()
    surface_point=trimesh.sample.sample_surface(tri_mesh,100000)[0] # Example surface points: 100000

    nss_samples=trimesh.sample.sample_surface(tri_mesh,500000)[0] # Example nss_samples: 500000
    nss_samples=nss_samples+np.random.randn(nss_samples.shape[0],3)*1*0.07

    resolution = 64
    vol_points = (np.random.random((500000, 3)) - 0.5) * 2 # Example vol_points: 500000

    nss_sdf=mesh_to_sdf.mesh_to_sdf(tri_mesh,nss_samples)
    grid_sdf=mesh_to_sdf.mesh_to_sdf(tri_mesh,vol_points)
    grid_label=(grid_sdf<0).astype(bool)
    near_label=(nss_sdf<0).astype(bool)
    os.makedirs(occ_save_dir,exist_ok=True)
    os.makedirs(pcd_save_dir,exist_ok=True)
    np.savez_compressed(occ_save_path,vol_points=vol_points.astype(np.float32),
                        vol_label=grid_label,near_points=nss_samples.astype(np.float32),
                        near_label=near_label)
    point_cloud_savepath=os.path.join(pcd_save_dir,f"{id}.npz")
    np.savez_compressed(point_cloud_savepath,points=surface_point.astype(np.float32))
    



def process_id(id):
    occ_save_dir = f"/xxx/crown_occ"
    pcd_save_dir = f"/xxx/crown_4_pointcloud"
    
    mesh_path = f"/xxx/tooth_crown_watertight/{id}.obj"
    process_object(mesh_path=mesh_path, id=id, occ_save_dir=occ_save_dir,pcd_save_dir=pcd_save_dir)
    print(f"Finished processing {id}")



def main():
    
    display = Display(visible=0, size=(1024, 768))
    display.start()
    
    id_list = os.listdir("/xxx/tooth_crown_watertight")
    id_list.sort(key=sort_key)
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for id_name in id_list[992:]:
            futures.append(executor.submit(process_id, id_name[:-4]))

    for future in as_completed(futures):
        pass
    
    display.stop()

if __name__ == "__main__":
    main()


