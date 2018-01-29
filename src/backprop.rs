use neuron::Neuron;

#[derive(Debug)]
struct BPNN {
    shape: Vec<usize>,
    layers: Vec<Vec<Neuron>>
}

impl BPNN {
    /* First index of shape is the number of inputs, the last index is the number of outputs.
       Everything index in between is a layer, where the number is the number of neurons in
       that layer. */
    pub fn new(shape: Vec<usize>) -> BPNN {
        let mut layers = Vec::new();
        for i in 1..shape.len() {
            let mut layer = Vec::new();
            for _ in 0..shape[i] {
                layer.push(Neuron::new(shape[i - 1]));
            }
            layers.push(layer);
        }

        BPNN {
            shape,
            layers
        }
    }

    pub fn train(&mut self, inputs: &Vec<f32>, expected: &Vec<f32>, lrate: f32) {
        //let mut sum_error = 0.0; // For logging error //
        let _ = self.feed_forward(inputs);
        /*for i in 0..expected.len() { ///////////////////
            sum_error += (expected[i] - outputs[i]).powi(2);
        }*////////////////////////////////////////////////
        self.back_propagate(expected);
        self.update_weights(inputs, lrate);
    }

    pub fn feed_forward(&mut self, inputs: &Vec<f32>) -> Vec<f32> {
        let mut feed = inputs.clone();
        for mut layer in self.layers.iter_mut() {
            let mut outputs = Vec::new();
            for mut neuron in layer.iter_mut() {
                outputs.push(neuron.fire(&feed));
            }
            feed = outputs;
        }
        feed
    }

    fn back_propagate(&mut self, expected: &Vec<f32>) {
        let num_layers = self.layers.len();
        for i in (0..num_layers).rev() {
            let mut errors = Vec::new();

            if i != num_layers - 1 {
                for j in 0..self.layers[i].len() {
                    let mut error = 0.0;
                    for neuron in &self.layers[i + 1] {
                        error += neuron.weight(j) * neuron.delta();
                    }
                    errors.push(error);
                }
            } else {
                for j in 0..self.layers[i].len() {
                    let delta = expected[j] - self.layers[i][j].output();
                    errors.push(delta);
                }
            }

            for j in 0..self.layers[i].len() {
                let derivative = self.layers[i][j].derivative();
                self.layers[i][j].set_delta(errors[j] * derivative);
            }
        }
    }

    fn update_weights(&mut self, inputs: &Vec<f32>, lrate: f32) {
        for i in 0..self.layers.len() {
            let mut feed = Vec::new();
            if i != 0 {
                for neuron in self.layers[i - 1].iter() {
                    feed.push(neuron.output());
                }
            } else {
                feed = inputs.clone();
            }

            for mut neuron in self.layers[i].iter_mut() {
                for j in 0..feed.len() {
                    let delta = neuron.delta();
                    neuron.adjust_weight(j, lrate * delta * feed[j]);
                }
                let bias_idx = neuron.bias_idx();
                let delta = neuron.delta();
                neuron.adjust_weight(bias_idx, lrate * delta);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! float_to_bool {
        ($x:expr) => (if $x >= 0.5 { true } else { false })
    }

    const INPUTS: [[f32; 2]; 4] = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ];

    #[test]
    fn new_network_shape() {
        let shape = vec![2, 2, 1];
        let nn = BPNN::new(vec![2, 2, 1]);

        // Check number of layers:  The first index are inputs NOT nodes, so there should be two layers.
        assert_eq!(2, nn.layers.len());

        // Check number of nodes in each layer.
        for i in 1..nn.layers.len() {
            assert_eq!(shape[i], nn.layers[i - 1].len());
        }
    }

    #[test]
    fn feed_forward() {
        let mut nn = BPNN::new(vec![2, 2, 1]);
        let outputs = nn.feed_forward(&vec![0.1, 1.0]);
        assert_ne!(outputs, [0.0]);
    }

    #[test]
    fn back_propagate() {
        let mut nn = BPNN::new(vec![2, 2, 1]);
        let _ = nn.feed_forward(&vec![0.1, 1.0]);
        let output_delta_before = nn.layers[1][0].delta();
        nn.back_propagate(&vec![0.0]);
        assert_ne!(output_delta_before, nn.layers[1][0].delta());
    }

    #[test]
    fn update_weights() {
        let mut nn = BPNN::new(vec![2, 2, 1]);
        let _ = nn.feed_forward(&vec![0.1, 1.0]);
        nn.back_propagate(&vec![0.0]);
        let weight_before = nn.layers[1][0].weight(0);
        let bias_before = nn.layers[1][0].weight(2);
        nn.update_weights(&vec![0.1, 1.0], 0.2);
        assert_ne!(weight_before, nn.layers[1][0].weight(0));
        assert_ne!(bias_before, nn.layers[1][0].weight(2));
    }

    #[test]
    fn logical_xor() {
        let mut nn = BPNN::new(vec![2, 3, 1]);
        let expected = vec![
            vec![0.0],
            vec![1.0],
            vec![1.0],
            vec![0.0]
        ];

        for _ in 0..2500 {
            for j in 0..INPUTS.len() {
                nn.train(&INPUTS[j].to_vec(), &expected[j], 0.2);
            }
        }

        for i in 0..INPUTS.len() {
            let outputs = nn.feed_forward(&INPUTS[i].to_vec());
            assert_eq!(float_to_bool!(expected[i][0]), float_to_bool!(outputs[0]));
        }
    }

    #[test]
    fn logical_xnor() {
        let mut nn = BPNN::new(vec![2, 3, 1]);
        let expected = vec![
            vec![1.0],
            vec![0.0],
            vec![0.0],
            vec![1.0]
        ];

        for _ in 0..2500 {
            for j in 0..INPUTS.len() {
                nn.train(&INPUTS[j].to_vec(), &expected[j], 0.2);
            }
        }

        for i in 0..INPUTS.len() {
            let outputs = nn.feed_forward(&INPUTS[i].to_vec());
            assert_eq!(float_to_bool!(expected[i][0]), float_to_bool!(outputs[0]));
        }
    }
}
