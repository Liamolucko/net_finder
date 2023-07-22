/// An iterator which yields all of the combinations of choices from each of the
/// input lists.
///
/// So, if you input a list of 3 lists of choices, this will yield `Vec`s 3
/// elements long where the first element is a choice from the first, the second
/// is a choice from the second, and the third is a choice from the third. And
/// it'll yield all possible `Vec`s like that.
pub struct Combinations<'a, T> {
    /// The choices for each component of the output combinations.
    choices: &'a [Vec<T>],
    /// The index of which choice we're going to yield next for each component.
    indices: Vec<usize>,
}

impl<'a, T> Combinations<'a, T> {
    pub fn new(choices: &'a [Vec<T>]) -> Self {
        Self {
            choices,
            indices: vec![0; choices.len()],
        }
    }
}

impl<T: Clone> Iterator for Combinations<'_, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self
            .indices
            .last()
            .is_some_and(|&index| index < self.choices.last().unwrap().len())
        {
            return None;
        }

        let combination = self
            .indices
            .iter()
            .enumerate()
            .map(|(index, &choice)| self.choices[index][choice].clone())
            .collect();

        for (index, choice) in self.indices.iter_mut().enumerate() {
            *choice += 1;
            if *choice >= self.choices[index].len() && index != self.choices.len() - 1 {
                // Carry over to the next index by resetting this to 0 and continuing to loop.
                *choice = 0;
            } else {
                // If we aren't carrying, break;
                break;
            }
        }

        Some(combination)
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::Combinations;

    #[test]
    fn combinations() {
        let choices = [vec![1, 2], vec![3, 4], vec![5, 6]];
        assert_eq!(
            Combinations::new(&choices).collect::<Vec<_>>(),
            [
                vec![1, 3, 5],
                vec![2, 3, 5],
                vec![1, 4, 5],
                vec![2, 4, 5],
                vec![1, 3, 6],
                vec![2, 3, 6],
                vec![1, 4, 6],
                vec![2, 4, 6]
            ]
        );
    }
}
