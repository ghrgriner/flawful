# Summary

This package has functionality for maintaining and managing language
learning flashcards and associated audio files. The audio syntax
generated is compatible with that used by Anki, a commonly used
flashcard software available on computers and smart phones. A git
repository for this package is located
[on github](https://github.com/ghrgriner/flawful), with additional
documentation in
[the wiki](https://github.com/ghrgriner/flawful/wiki) of the repository.

The primary objective of sharing this package and documentation is to
aid users who ask the question “How can I use spaced-repetition
flashcard software to learn a foreign language?” We present
suggestions with example code for the case of an English native-speaker
studying German, but the code is designed for use with arbitrary
languages (although users might need to write their own functions
that can be input to the package functions to parse their input).

This package does not contain pre-made flashcard decks for studying a
language. Making flashcard decks from scratch can be a tedious task,
and, in our opinion, this is not always useful for learning. Therefore,
we do not discourage users from using pre-made decks if they find
some suitable for their language. However, this code might still
be useful for such users if they want to build a deck to manage or
supplement their pre-made deck (for example, to learn vocabulary for
their particular profession). This documentation may also be useful
for users who have started with a pre-made deck that they now feel
could be improved, as it provides concrete suggestions and code for
such improvements.

# Functionality

This section uses terms defined on the [Background and Definitions](https://github.com/ghrgriner/flawful/wiki/Background-and-Definitions)
page of the wiki.

At a high level, the most important features are integrating audio
information into the deck, integrating information from multiple
reference lists into the deck (for example, to assign each note a
chapter for study), and creating tags useful for organizing studying.

Functionality applicable to most languages:

* Add audio files to a deck and print the keys for the tokens where
no matching audio was found. User-defined functions can be used to
filter the keys automatically (e.g., by assigned chapter or excluding
keys with spaces or special characters) so that the number printed
out for manual review is less. Similarly, the audio files not matched
to a token in the input notes can be printed.

* Define classes representing reference lists and provide example
code for populating objects of these classes. An important feature
is that each entry in a reference list can have a chapter assigned,
and the reference itself can have a chapter offset assigned. Once
all reference lists are in a standard format, package functions can
be used to perform common tasks, such as: (1) calculating the final
chapter for each note as the minimum (chapter + offset) for any word
on the note, (2) generating HTML tags that can be formatted in the
flashcard program to indicate the reference list a token is from
(e.g., A1=blue, A2=green), (3) printing words from each reference
list not found in the input notes, (4) generating one-way or multi-way
frequency counts of the number of headwords found in the intersections
of arbitrary combinations of reference lists.

* Example code is provided for loading reference lists from text files
in various formats including: words grouped by chapter instead of
chapter being indicated on each individual line, (2) chapters parsed
from individual lines; (3) word, chapter, and example sentences in a
flat file in format that might have been exported from a spreadsheet
program.

* Automatically generate the target number of words to be given in
the ‘primary’ answer. This includes support for considering words
marked with ‘°’ as optional, so a prompt of ‘2/3’ indicates 2
words required, and a third optional. This also supports the indication
of the number of target answers in up to two dialect fields (e.g.,
Austrian and/or Swiss-German), so that a prompt of ‘1 + A:1’
indicates the expected answer includes one word used in Germany and
one used in Austria.

Functionality specific to German (although applicable with slight
modifications to other languages):

* Users can indicate long/short vowels in designated fields of the
input notes using ‘[[v]]’ and ‘[v]’, respectively, and the
program will convert these in the output to underlining or a letter
with an under-dot.

* We recommend that German flashcards have one field with the singular
form of the noun(s) for notes and another for the plural form(s). This
second field can also be used to hold the three principal parts of the
verb (or an abbreviation indicating the conjugation class), whereas
the the first field holds the infinitive. The package has a function
to check that this second field has the expected number of tokens
in these cases, and also checks (if the user follows a convention)
whether sufficient information is present in the second field to
indicate the separability of a prefix for those verbs with prefixes
that can be separable or inseparable.

* An example using about 40 German notes is distributed
with the package using vocabulary and audio files taken
from the open-source textbook [Deutsch im Blick (2nd edition,
2017)](https://coerll.utexas.edu/dib/introduction.php) and its online
companion grammar [Grimm Grammar](https://coerll.utexas.edu/gg/). (See
package LICENSE.txt for licensing details.) The example also
illustrates how to import audio files named using the most common
naming convention for German audio from the German Wiktionary. The
input notes file for this example illustrates many recommendations we
believe to be useful when creating flashcards. Documentation for the
input notes file along with general recommendations for columns to
include when making flashcards will be made available on the project
wiki at the git repository listed above. There is also documentation
for each note in the input file in the `comments` field that lists the
point(s)-of-emphasis for a given note in the example. Users interested
in more material from Deutsch im Blick and Grimm Grammar can refer to
[this unofficial Anki deck](https://ankiweb.net/shared/info/773249133),
which has about 2000 notes and 2500 audio files, or to the websites
linked above.

# Example

An example program is contained in the package at
`src/flawful/examples/example1.py`.
See the wiki for details on the structure of the
input notes file associated with the [latest version of this file](https://github.com/ghrgriner/flawful/blob/main/src/flawful/examples/example1.py).

# Other Resources

We have also authored the `wikwork` package that is designed to
retrieve word and audio file information from Wiktionary.

