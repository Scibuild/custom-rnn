use ndarray::{Array1, Array2, Array};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

use serde::{Deserialize, Serialize};

// for when you come back next year:
//
// # Feeding forward
//
// Each layer is a Matrix where the rows are the neurons
//      and the values are the weights
//
//  [w1->1  w2->1  w3->1 ...]
//  [w1->2  w2->2  w3->2 ...]
//  [w1->3  w2->3  w3->3 ...]
//  [ ...    ...    ...  ...]
//
//  The Biases for each layer are a column vector for each neuron
//  [ b1 ]
//  [ b2 ]
//  [ b3 ]
//  [ .. ]
//
//  to calculate the activation for that layer of neurons, multiply the weights
//      with the previous layer's activations and add the biases, then sigmoid the whole thing
//
//  [w1->1  w2->1  w3->1 ...] [ a1 ]     [ b1 ]     [ z1 ]
//  [w1->2  w2->2  w3->2 ...] [ a2 ]  +  [ b2 ]  =  [ z2 ] = Z
//  [w1->3  w2->3  w3->3 ...] [ a3 ]     [ b3 ]     [ z3 ]
//  [ ...    ...    ...  ...] [ .. ]     [ .. ]     [ .. ]
//
//  σ(Z) = A
//
//  remember that multiplication by matrix with a vector transforms it from having
//      the same number of values as the matrix has columns into the same number of
//      values as the matrix has rows
//
//  [ a1·w1->1 + a2·w2->1 + ... + b1 ]
//  [ a1·w1->2 + a2·w2->2 + ... + b2 ] = Z
//  [ a1·w1->3 + a2·w2->3 + ... + b3 ]
//  [ ... ... ... ... ... ... ... ...]
//
//  # Backpropogation
//
//  Look in the annotated back_propogate function
//

type WeightMatrix = Array2<f64>;
type Weights = Vec<WeightMatrix>;
type ActivationVector = Array1<f64>;
type Activations = Vec<Array1<f64>>;
type Biases = Vec<BiasVector>;
type BiasVector = Array1<f64>;

pub mod util {
    use super::ActivationVector;
    use ndarray::{Array1, Array2, Axis};

    pub fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (std::f64::consts::E.powf(-z)))
    }

    pub fn sigmoid_vector(v: &ActivationVector) -> ActivationVector {
        v.mapv(sigmoid)
    }

    // pub fn cost(output: &ActivationVector, desired: &ActivationVector) -> f64 {
    //     // output.iter().zip(desired.iter()).fold(0f64, |sum, (&o, &d)| sum + (d-o).powf(2.0))
    //     (desired - output).mapv(|f| f.powi(2)).sum()
    // }

    // C = cost function
    // a = output vector
    //      ∂C
    // C' = --
    //      ∂a
    // which means that you take the partial derivative of C with respect to a small change in a
    // given that the other input (desired output) remains constant. This tells you how to change a
    // in order to increase C(a,d), which is not what we want to do, which is decrease the cost. If
    // C' is positive then increasing a will increase C, if C' is negative then increasing a will
    // decrease C. now we know how to change the outputs to reduce the cost function output.
    //
    // Working out:
    //
    // summing any vector = multiplication with matrix like [1 1 1 1 ... ]
    // before summing, each value in cost vector = (d-a)²
    // summing rule for derivates means derivative of sums = sum of derivates
    // ∴ cost is equally impacted by each thingy, um derivative is a vector for each value of a
    //
    //  ∂(d-a)²                   ∂C ∂u   ∂(u²)        ∂(d-a)
    //  ------- ; let u = (d-a) ; --·-- ; ----- = 2u ; ------ = -1
    //    ∂a                      ∂u ∂a    ∂u            ∂a
    //    ∂C ∂u
    //  ∴ --·-- = 2u·-1 = -2(d-a) = 2(a-d)
    //    ∂u ∂a
    //
    //  ∂C
    //  -- = 2(a-d)
    //  ∂a
    //
    //  some people leave of the 2, but my math tells me it exists and i want this readable
    //
    pub fn cost_derivative(
        output: &ActivationVector,
        desired: &ActivationVector,
    ) -> ActivationVector {
        (output - desired).mapv(|x| x * 2f64)
    }

    pub fn sigmoid_derivative_of_output(a: f64) -> f64 {
        a * (1.0 - a)
    }
    pub fn sigmoid_derivative_of_output_vector(a: &ActivationVector) -> ActivationVector {
        a.mapv(sigmoid_derivative_of_output)
    }

    // https://en.wikipedia.org/wiki/Outer_product
    pub fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
        let column_a: Array2<f64> = a.clone().insert_axis(Axis(1));
        let row_b: Array2<f64> = b.clone().insert_axis(Axis(0));
        column_a.dot(&row_b)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Network {
    // neurons: Layers,    // Layer number, neuron number, (wheights, biases)
    // num_layers: usize,              // Number of Layers
    layer_sizes: Vec<usize>, // A vector of sizes of all the arrays
    weights: Weights,
    biases: Biases,
}

impl Network {
    pub fn new(layers: Vec<usize>) -> Network {
        let mut weights: Weights = Vec::with_capacity(layers.len() - 1);
        let mut biases: Biases = Vec::with_capacity(layers.len() - 1);

        for i in 0..layers.len()-1 {
            let randweights: Array2<f64> =
                Array2::random((layers[i+1], layers[i]), StandardNormal);
            let randbiases: Array1<f64> = Array1::random((layers[i+1],), StandardNormal);
            weights.push(randweights);
            biases.push(randbiases);
        }

        Network {
            weights: weights,
            biases: biases,
            layer_sizes: layers,
        }
    }

    pub fn feed_forward(&self, input: &ActivationVector) -> ActivationVector {
        // self.neurons.iter().fold(input, |acc, x| calculate_layer(x, acc))
        // does the same thing but nicer to read
        // println!("================= FEEDING FORWARD ===============");
        // println!("       ==========   layer 0       ========       ");
        let mut activations = input.clone();
        // println!("activations = {:?}",activations);
        for i in 0..self.weights.len() {
            // println!("\n\n       ==========   layer {}       ========       ", i+1);
            // println!("self.weight.len() = {}", self.weights.len());
            // println!("self.biases.len() = {}", self.biases.len());
            let z = self.weights[i].dot(&activations) + &self.biases[i];
            // println!("z = {:?}",z);
            activations = util::sigmoid_vector(&z);
            // println!("activations = {:?}",activations);
        }
        activations
    }

    pub fn back_propogate(
        &mut self,
        input: &ActivationVector,
        target: &ActivationVector,
    ) -> (Weights, Biases) {

        // feed forward but storing state
        let mut activations: Activations = Vec::with_capacity(self.layer_sizes.len());
        let mut activation = input.clone(); //TODO: decide if clone is appropriate here
        activations.push(activation.clone());

        let mut zs = Vec::with_capacity(self.layer_sizes.len() - 1);


        for i in 0..self.weights.len() {
            let z = self.weights[i].dot(&activation) + &self.biases[i];
            activation = util::sigmoid_vector(&z);
            zs.push(z);
            activations.push(activation.clone());
        }
        // now magic

        // so δ of a layer is the vector of the partial derivatives of the cost function with
        // respect to a change in the z value of those neurons (pre-squishification)
        //
        // ∇bias = δ    {the derivative of the biases vector is the derivative of the whole neuron,
        //               z = wa + b, holding wa constant leaves just b for derivative}
        //
        // ∇weights = δ·aᵀ
        //
        //  [ δ1 ]                          [ a1·δ1  a2·δ1  a2·δ1 ... ]
        //  [ δ2 ] . [ a1  a2  a3 ... ]  =  [ a1·δ2  a2·δ2  a2·δ2 ... ]
        //  [ δ3 ]                          [ a1·δ3  a2·δ3  a2·δ3 ... ]
        //  [ .. ]                          [  ...    ...    ...  ... ]
        //
        //              {   some neuron n = w1·a1+w2·a2+w3·a3... + b
        //                  δ for that neuron ( δᵢ) is ∂C/∂n
        //                  in order to get ∂C/∂w for weight j = ∂C/∂n * ∂n/∂wⱼ
        //                  ∂n/∂wⱼ= ∂/∂wⱼ(w1·a1+w2·a2+w3·a3... + b) = aⱼ
        //                  ∴ ∂C/∂wⱼ= δᵢ·aⱼ
        //                  the vector matrix stuff just makes it easy to calculate for every
        //                  neuron
        //
        //                  remember that the activations here are from the previous layer
        //              }
        //

        let mut nabla_biases: Biases = Vec::with_capacity(self.biases.len()); // ∇bias[]
        let mut nabla_weights: Weights = Vec::with_capacity(self.weights.len()); // ∇bias[]
        let num_layers = activations.len();

        // δ = ∂C/∂z = ∂C/∂a·∂a/∂z
        // ∂a/∂z = σ'(z) = σ(z)*(1-σ(z)) = a*(1-a)
        let mut delta = util::cost_derivative(&activation, target)
            * util::sigmoid_derivative_of_output_vector(&activations[num_layers - 1]);
        // we build up these Vecs in reverse order and then reverse them at the end
        nabla_biases.push(delta.clone());
        nabla_weights.push(util::outer_product(&delta, &activations[num_layers - 2]));

        // first one was special because cost function stuff, now we just loop through the rest of
        // the layers
        //
        // δ = ∂C/∂zᵢ= ∂C/∂a·∂a/∂z₀·Π(∂zⱼ/∂aᵢ·∂aᵢ/∂zᵢ)
        //
        // ∂aᵢ/∂zᵢ= σ'
        //
        // ∂zⱼ/∂aᵢ= ∂/∂aᵢ(wᵢaᵢ+bᵢ)
        //
        //  [w1->1  w2->1  w3->1 ...] [ a1 ]     [ b1 ]     [ z1 ]
        //  [w1->2  w2->2  w3->2 ...] [ a2 ]  +  [ b2 ]  =  [ z2 ] = Z
        //  [w1->3  w2->3  w3->3 ...] [ a3 ]     [ b3 ]     [ z3 ]
        //  [ ...    ...    ...  ...] [ .. ]     [ .. ]     [ .. ]
        //
        //                [w1->1  w2->1  w3->1 ...] [ a1 ]   [ a1·w1->1 + a2·w2->1 + ... ]
        // ∂zⱼ/∂aᵢ= ∂/∂aᵢ [w1->2  w2->2  w3->2 ...] [ a2 ] = [ a1·w1->2 + a2·w2->2 + ... ]
        //                [w1->3  w2->3  w3->3 ...] [ a3 ]   [ a1·w1->3 + a2·w2->3 + ... ]
        //                [ ...    ...    ...  ...] [ .. ]   [ ... ...    ... ...    ... ]
        //
        // ∂z/∂an = [wn->1  wn->2  wn->3 ...]
        //
        // stack all of these on top of each other and you get:
        //
        //           [w1->1  w1->2  w1->3 ...]
        // ∂zⱼ/∂aᵢ = [w2->1  w2->2  w2->3 ...] = wᵀ
        //           [w3->1  w3->2  w3->3 ...]
        //           [ ...    ...    ...  ...]
        //
        // delta is always a column vector of zds so what happens is that this gets multiplied with
        // the previous layer's delta to describe how each weight impacts not an individual neuron,
        // but all of them combined
        //
        //   its basically the chain rule a lot
        //   each row is the sum of the effects that that neuron has on the ones in the next layer,
        //   times the effect that neuron has on the final cost function
        //
        //   [ ∂C/∂z1ⱼ·∂z1ⱼ/∂a1ᵢ+ ∂C/∂z2ⱼ·∂z2ⱼ/∂a1ᵢ+ ∂C/∂z3ⱼ·∂z3ⱼ/∂a1ᵢ ... ]
        //   [ ∂C/∂z1ⱼ·∂z1ⱼ/∂a2ᵢ+ ∂C/∂z2ⱼ·∂z2ⱼ/∂a2ᵢ+ ∂C/∂z3ⱼ·∂z3ⱼ/∂a2ᵢ ... ] =
        //   [ ∂C/∂z1ⱼ·∂z1ⱼ/∂a3ᵢ+ ∂C/∂z2ⱼ·∂z2ⱼ/∂a3ᵢ+ ∂C/∂z3ⱼ·∂z3ⱼ/∂a3ᵢ ... ]
        //   [  ...      ...        ...     ...       ...       ...    ... ]
        //
        //   [ δ1·w1->1 + δ2·w1->2 + δ3·w1->3 ... ]   [w1->1  w1->2  w1->3 ...] [ δ1 ]
        //   [ δ1·w2->1 + δ2·w2->2 + δ3·w2->3 ... ] = [w2->1  w2->2  w2->3 ...] [ δ2 ] = wᵀδ
        //   [ δ1·w3->1 + δ2·w3->2 + δ3·w3->3 ... ]   [w3->1  w3->2  w3->3 ...] [ δ3 ]
        //   [ ... ...    ... ...    ... ...  ... ]   [ ...    ...    ...  ...] [ .. ]
        //
        //  finally, combine the thing from the top with this to get:
        //
        //  δᵢ= wᵀδⱼ⊙ σ'( aᵢ)   {⊙ is element-wise multiplication for vectors, just * in ndarray}
        //                      {wᵀis for wⱼbecause its associated with previous layer in code}
        //

        for l in 2..num_layers {
            let sp = util::sigmoid_derivative_of_output_vector(&activations[num_layers - l]);
            delta = self.weights[num_layers - l].t().dot(&delta) * sp;
            nabla_biases.push(delta.clone());
            nabla_weights.push(util::outer_product(
                &delta,
                &activations[num_layers - l - 1],
            ));
        }

        // reverse nabla Vecs
        nabla_weights.reverse();
        nabla_biases.reverse();

        (nabla_weights, nabla_biases)
    }

    pub fn update_in_batch(&mut self, training_data: &Vec<(ActivationVector, ActivationVector)>, eta: f64) -> &mut Self {

        let mut nabla_weights: Weights = Vec::with_capacity(self.weights.len());
        for w in &self.weights {
            nabla_weights.push(Array::zeros(w.raw_dim()));
        }

        let mut nabla_biases: Biases = Vec::with_capacity(self.biases.len());
        for b in &self.biases {
            nabla_biases.push(Array::zeros(b.raw_dim()));
        }

        for (input, target) in training_data {
            let (delta_nabla_weights, delta_nabla_biases) = self.back_propogate(input, target);
            // sums up all the thingos from the back propogations to be averaged later
            for i in 0..nabla_weights.len() {
                nabla_weights[i].scaled_add(1.0, &delta_nabla_weights[i]);
                nabla_biases[i].scaled_add(1.0, &delta_nabla_biases[i]);
            }

        }

        let scale_factor = -eta/(training_data.len() as f64);
        for i in 0..self.weights.len() {
            self.weights[i].scaled_add(scale_factor, &nabla_weights[i]);
            self.biases[i].scaled_add(0.1*scale_factor, &nabla_biases[i]);
        }

        self
    }

}
