use std::fmt;

pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Tensor {
            data: vec![0.0; size],
            shape,
        }
    }

    fn get_index(&self, indices: &[usize]) -> usize {
        indices
            .iter()
            .zip(self.shape.iter().skip(1).chain(std::iter::once(&1)))
            .fold(0, |acc, (&idx, &dim)| acc * dim + idx)
    }

    fn set(&mut self, indices: &[usize], value: f64) {
        let idx = self.get_index(indices);
        self.data[idx] = value;
    }

    fn get(&self, indices: &[usize]) -> f64 {
        let idx = self.get_index(indices);
        self.data[idx]
    }

    // Beispiel für eine Matrixmultiplikation mit festem f64-Typ
    fn mul_2d(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape[1], other.shape[0],
            "Incompatible shapes for multiplication"
        );

        let (m, n, p) = (self.shape[0], self.shape[1], other.shape[1]);
        let mut output = Tensor::new(vec![m, p]);

        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += self.get(&[i, k]) * other.get(&[k, j]);
                }
                output.set(&[i, j], sum);
            }
        }

        output
    }

    fn mul_1d(&self, other: &Tensor) -> f64 {
        assert_eq!(self.shape[0], other.shape[0], "Wrong input shape");
        let mut sum = 0.0;

        for (i, j) in self.data.iter().zip(other.data.iter()) {
            sum += *i * *j;
        }

        sum
    }

    // TODO: Fix this.
    // Probably indexing is wrong.
    pub fn mul_3d(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape[0], other.shape[0], "Batch sizes must match");
        assert_eq!(self.shape[2], other.shape[1], "Inner dimensions must match");

        let (batch, m, n) = (self.shape[0], self.shape[1], self.shape[2]);
        let p = other.shape[2];
        let mut result = Tensor::new(vec![batch, m, p]);

        for b in 0..batch {
            for i in 0..m {
                for j in 0..p {
                    let mut sum = 0.0;
                    for k in 0..n {
                        // Debugging der Indizes
                        let a_val = self.get(&[b, i, k]);
                        let b_val = other.get(&[b, k, j]);
                        sum += a_val * b_val;

                        // Optional: Debug-Ausgabe zur Verfolgung der Werte
                        println!(
                            "Batch {}, A[{}, {}, {}] * B[{}, {}, {}] = {} * {}",
                            b, i, k, j, b, k, j, a_val, b_val
                        );
                    }
                    result.set(&[b, i, j], sum);
                }
            }
        }

        result
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.shape.len() {
            1 => {
                write!(f, "[")?;
                for (i, val) in self.data.iter().enumerate() {
                    write!(f, "{}", val)?;
                    if i < self.data.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "]")
            }
            2 => {
                let rows = self.shape[0];
                let cols = self.shape[1];
                for i in 0..rows {
                    write!(f, "[")?;
                    for j in 0..cols {
                        write!(f, "{}", self.data[i * cols + j])?;
                        if j < cols - 1 {
                            write!(f, ", ")?;
                        }
                    }
                    if i < rows - 1 {
                        write!(f, "]\n")?;
                    } else {
                        write!(f, "]")?;
                    }
                }
                Ok(())
            }
            _ => {
                write!(
                    f,
                    "Tensor with shape {:?} and data {:?}",
                    self.shape, self.data
                )
            }
        }
    }
}
