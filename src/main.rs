use simple_nn::utils::tensor::Tensor;
fn main() {
    let a = Tensor {
        data: vec![
            1.0, 2.0, // Batch 1, Row 1
            3.0, 4.0, // Batch 1, Row 2
            5.0, 6.0, // Batch 1, Row 3
            7.0, 8.0, // Batch 2, Row 1
            9.0, 10.0, // Batch 2, Row 2
            11.0, 12.0, // Batch 2, Row 3
        ],
        shape: vec![2, 3, 2], // 2 Batches, 3 Rows, 2 Columns
    };

    let b = Tensor {
        data: vec![
            1.0, 2.0, // Batch 1, Row 1
            3.0, 4.0, // Batch 1, Row 2
            5.0, 6.0, // Batch 2, Row 1
            7.0, 8.0, // Batch 2, Row 2
        ],
        shape: vec![2, 2, 2], // 2 Batches, 2 Rows, 2 Columns
    };

    let result = a.mul_3d(&b);

    println!("Result: {}", result);
}
