use ndarray::{Array1, Array2};
use sprs::{CsMat, TriMat};

/// Generates a structured triangular mesh on a rectangular domain
pub struct Mesh {
    pub nodes: Array2<f64>,      // (num_nodes, 2) - node coordinates
    pub elements: Array2<usize>,  // (num_elements, 3) - triangle node indices
    pub num_nodes: usize,
    pub num_elements: usize,
}

impl Mesh {
    pub fn new(lx: f64, ly: f64, nx: usize, ny: usize) -> Self {
        let num_nodes = (nx + 1) * (ny + 1);
        let num_elements = 2 * nx * ny;
        
        // Generate node coordinates
        let mut nodes = Array2::zeros((num_nodes, 2));
        let dx = lx / (nx as f64);
        let dy = ly / (ny as f64);
        
        for j in 0..=ny {
            for i in 0..=nx {
                let node_id = j * (nx + 1) + i;
                nodes[[node_id, 0]] = i as f64 * dx;
                nodes[[node_id, 1]] = j as f64 * dy;
            }
        }
        
        // Generate elements (triangles)
        let mut elements = Array2::zeros((num_elements, 3));
        let mut elem_id = 0;
        
        for j in 0..ny {
            for i in 0..nx {
                let n1 = j * (nx + 1) + i;
                let n2 = n1 + 1;
                let n3 = (j + 1) * (nx + 1) + i;
                let n4 = n3 + 1;
                
                // Triangle 1: (n1, n2, n4)
                elements[[elem_id, 0]] = n1;
                elements[[elem_id, 1]] = n2;
                elements[[elem_id, 2]] = n4;
                elem_id += 1;
                
                // Triangle 2: (n1, n4, n3)
                elements[[elem_id, 0]] = n1;
                elements[[elem_id, 1]] = n4;
                elements[[elem_id, 2]] = n3;
                elem_id += 1;
            }
        }
        
        Mesh {
            nodes,
            elements,
            num_nodes,
            num_elements,
        }
    }
    
    /// Get boundary edges for a structured mesh
    pub fn boundary_edges(&self, nx: usize, ny: usize) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        
        // Bottom edge (j=0)
        for i in 0..nx {
            edges.push((i, i + 1));
        }
        
        // Top edge (j=ny)
        let offset = ny * (nx + 1);
        for i in 0..nx {
            edges.push((offset + i, offset + i + 1));
        }
        
        // Left edge (i=0)
        for j in 0..ny {
            let n1 = j * (nx + 1);
            let n2 = (j + 1) * (nx + 1);
            edges.push((n1, n2));
        }
        
        // Right edge (i=nx)
        for j in 0..ny {
            let n1 = j * (nx + 1) + nx;
            let n2 = (j + 1) * (nx + 1) + nx;
            edges.push((n1, n2));
        }
        
        edges
    }
}

/// Compute element matrices for a linear triangular element
pub fn element_matrices(coords: &[[f64; 2]; 3]) -> (Array2<f64>, Array2<f64>, f64) {
    let (x1, y1) = (coords[0][0], coords[0][1]);
    let (x2, y2) = (coords[1][0], coords[1][1]);
    let (x3, y3) = (coords[2][0], coords[2][1]);
    
    // Calculate area
    let area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)).abs();
    
    // Gradient coefficients
    let b = [y2 - y3, y3 - y1, y1 - y2];
    let c = [x3 - x2, x1 - x3, x2 - x1];
    
    // Stiffness matrix
    let mut ke = Array2::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            ke[[i, j]] = (b[i] * b[j] + c[i] * c[j]) / (4.0 * area);
        }
    }
    
    // Mass matrix (consistent)
    let me = (area / 12.0) * Array2::from_shape_vec(
        (3, 3),
        vec![2.0, 1.0, 1.0,
             1.0, 2.0, 1.0,
             1.0, 1.0, 2.0]
    ).unwrap();
    
    (me, ke, area)
}

/// Assemble global mass and stiffness matrices
pub fn assemble_system(mesh: &Mesh) -> (CsMat<f64>, CsMat<f64>) {
    let n = mesh.num_nodes;
    let mut m_tri = TriMat::new((n, n));
    let mut k_tri = TriMat::new((n, n));
    
    for elem_idx in 0..mesh.num_elements {
        // Get element node coordinates
        let mut coords = [[0.0; 2]; 3];
        for i in 0..3 {
            let node_id = mesh.elements[[elem_idx, i]];
            coords[i][0] = mesh.nodes[[node_id, 0]];
            coords[i][1] = mesh.nodes[[node_id, 1]];
        }
        
        let (me, ke, _area) = element_matrices(&coords);
        
        // Assemble into global matrices
        for i in 0..3 {
            for j in 0..3 {
                let row = mesh.elements[[elem_idx, i]];
                let col = mesh.elements[[elem_idx, j]];
                m_tri.add_triplet(row, col, me[[i, j]]);
                k_tri.add_triplet(row, col, ke[[i, j]]);
            }
        }
    }
    
    (m_tri.to_csr(), k_tri.to_csr())
}

/// Compute boundary matrices for Robin boundary conditions
pub fn compute_boundary_matrices(
    mesh: &Mesh,
    edges: &[(usize, usize)],
    h_coeff: f64,
) -> (CsMat<f64>, Array1<f64>) {
    let n = mesh.num_nodes;
    let mut k_bound_tri = TriMat::new((n, n));
    let mut f_bound = Array1::zeros(n);
    
    for &(n1, n2) in edges {
        let x1 = mesh.nodes[[n1, 0]];
        let y1 = mesh.nodes[[n1, 1]];
        let x2 = mesh.nodes[[n2, 0]];
        let y2 = mesh.nodes[[n2, 1]];
        
        let length = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
        
        // 1D boundary mass matrix
        let me_1d = (length / 6.0) * Array2::from_shape_vec(
            (2, 2),
            vec![2.0, 1.0, 1.0, 2.0]
        ).unwrap();
        
        // 1D load vector
        let fe_1d = (length / 2.0) * Array1::from_vec(vec![1.0, 1.0]);
        
        // Assemble
        let indices = [n1, n2];
        for i in 0..2 {
            f_bound[indices[i]] += fe_1d[i] * h_coeff;
            for j in 0..2 {
                k_bound_tri.add_triplet(indices[i], indices[j], me_1d[[i, j]] * h_coeff);
            }
        }
    }
    
    (k_bound_tri.to_csr(), f_bound)
}
