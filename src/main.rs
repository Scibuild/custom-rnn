mod network;
use ndarray::Array1;
use network::Network;

use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

use rand;
use rand::prelude::*;

#[derive(PartialEq)]
enum CommandName {
    Run,
    Train,
    Debug,
    Test,
}

// #[derive(Deserialize)]
// struct Tweet {
// count: i32,
// hate_speech: i32,
// offensive_language: i32,
// neither: i32,
// class: usize,
// tweet: String
// }

const LENGTH_OF_MAX_WORD: usize = 15;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() == 1 {
        eprintln!("Please try {} <run|train|test> [file]", args[0]);
        return Ok(());
    }

    let cmd = match args[1].as_ref() {
        "run" => CommandName::Run,
        "train" => CommandName::Train,
        "debug" => CommandName::Debug,
        "test" => CommandName::Test,
        x => {
            eprintln!(
                "Not recognised command: {}\nTry: cargo run <run|train> [file]",
                x
            );
            return Ok(());
        }
    };

    let mut neural_network = if cmd != CommandName::Run {
        let layers = vec![27 * LENGTH_OF_MAX_WORD, 20, 2];
        Network::new(layers)
    } else {
        let f = File::open(&args[2])?;
        let mut buf_reader = BufReader::new(f);
        let mut file_contents = String::new();
        buf_reader.read_to_string(&mut file_contents)?;
        serde_json::from_str(&file_contents).unwrap()
    };

    match cmd {
        CommandName::Train => {

            let mut training_data = read_training_data(&args[3], &args[4])?;

            let mut rng = rand::thread_rng();
            training_data.shuffle(&mut rng);

            let batch_size = 50;
            let num_batches = training_data.len() / batch_size;
            let bar_length = 50;
            for group in 0..num_batches {
                let chars_filled = group * bar_length / num_batches;
                let completed_bar = "-".repeat(chars_filled);
                let incomplete_bar = " ".repeat(bar_length - chars_filled - 1);
                print!(
                    "Training: [{}>{}] {:.1}%\r",
                    completed_bar,
                    incomplete_bar,
                    group * 100 / num_batches
                );

                let batch = training_data[group * batch_size..(group + 1) * batch_size].to_vec();
                neural_network.update_in_batch(&batch, 1.0);
            }

            let serialized_network = serde_json::to_string(&neural_network).unwrap();
            std::fs::write(&args[2], serialized_network)?;
        }
        CommandName::Run => {
            println!("please enter some text:");
            let mut text = input("> ");
            loop {
                if text.trim().len() == 0 {
                    break;
                }
                let text_activation_array = format_text_for_learning(&text, LENGTH_OF_MAX_WORD);

                println!(
                    "\n\nOutput result: {}",
                    activation_array_to_string(neural_network.feed_forward(&text_activation_array))
                );
                text = input("> ");
            }
        }
        CommandName::Test => {
            let mut testing_data = read_training_data(&args[3], &args[4])?;
            let mut rng = rand::thread_rng();
            testing_data.shuffle(&mut rng);

            let mut score = 0;
            let mut trials = 0;
            for (input, output) in testing_data {
                let result = neural_network.feed_forward(&input);
                trials += 1;

                let which_result = result[0] > result[1];
                let which_correct = output[0] > output[1];

                if which_result == which_correct {
                    score += 1;
                }
                
                print!("{}/{}   {}%\r", score, trials, score * 100 / trials);
            }
            println!("{}/{}   {}%", score, trials, score * 100 / trials);
        }
        CommandName::Debug => {
        }
    }

    Ok(())
}

fn int_to_array(i: usize, max_i: usize) -> Array1<f64> {
    let mut e = Array1::zeros((max_i,));
    e[i] = 1.0;
    e
}

fn format_text_for_learning(text: &str, length: usize) -> Array1<f64> {
    let mut text_activation_vec: Vec<f64> = vec![0.0].repeat(27 * LENGTH_OF_MAX_WORD);

    text.trim()
        .to_lowercase()
        .chars()
        .filter(|c| ((*c as u8) < 123 && (*c as u8) > 96) || (*c as u8) == 32)
        .take(length)
        .enumerate()
        .for_each(|(i, c)| {
            let ascii_c = c as u8;
            let transformed_c = if ascii_c == 32 {
                26 as usize
            } else {
                (ascii_c - 97) as usize
            };
            text_activation_vec[i * 27 + transformed_c] = 1.0;
        });

    Array1::from(text_activation_vec)
}

fn read_training_data(file_name1: &str, file_name2: &str) -> std::io::Result<Vec<(Array1<f64>, Array1<f64>)>> {

    let lang1 = file_to_string(file_name1)?;
    let mut lang1_formatted = parse_for_language(lang1, 0, 2);

    let lang2 = file_to_string(file_name2)?;
    let mut lang2_formatted = parse_for_language(lang2, 1, 2);
    lang1_formatted.append(&mut lang2_formatted);


    Ok(lang1_formatted)
}


fn parse_for_language(
    lines: String,
    i: usize,
    num: usize,
) -> Vec<(Array1<f64>, Array1<f64>)> {
    lines.lines()
        .map(|line| {
            let language = int_to_array(i, num);
            let (word, _) = line.split_at(line.find(',').unwrap());
            let word_activations = format_text_for_learning(word, LENGTH_OF_MAX_WORD);
            (word_activations, language)
        })
        .collect()
}

fn file_to_string(name: &str) -> std::io::Result<String> {
    let data_file = File::open(name)?;
    let mut buf_data_file_reader = BufReader::new(data_file);
    let mut raw_data = String::new();
    buf_data_file_reader.read_to_string(&mut raw_data)?;
    Ok(raw_data)
}

fn input(prompt: &str) -> String {
    print!("{}", prompt);
    std::io::stdout().flush().unwrap();
    let mut text = String::new();
    std::io::stdin().read_line(&mut text).unwrap();
    text
}

fn activation_array_to_string(pvec: Array1<f64>) -> String {
    let mut string = String::new();

    let mut largest: f64 = 0.0;
    let mut largest_index: usize = 0;
    for (i, probability) in pvec.into_iter().enumerate() {
        if probability > &largest {
            largest = *probability;
            largest_index = i;
        }
        // final element of 27 float blocks
        if i % 27 == 26 {
            if largest_index == i || largest < 0.2 {
                string.push(' ');
            } else {
                string.push((largest_index % 27 + 97) as u8 as char)
            }
            largest = 0.0;
        }
    }

    string
}
