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

# Background and Definitions

## Program creates cards / notes

The purpose of the package is to provide tools for converting a
language learning ‘database’ stored in a flat file (e.g., which
might have been exported from a spreadsheet) into flashcards for
studying. More precisely, to use the terminology of Anki software, this
generates notes which can be used to generate one or more flashcard per
note in Anki, but in the rest of this documentation, we will use the
terms flashcards or notes to refer to a record in the generated output.

## Reference lists

The term ‘reference list’ refers to some list of words, typically
not created by the user. For example, one reference list might be a
list of A1 words from a testing authority, another reference list
might be a list of words from an A1 textbook or website, a third
reference list is a list of A2 words, etc... One of the purposes of
the program is to provide tools for merging the input notes with the
reference list(s) [for example, to check whether all words in a given
reference list are in an input file]. A reference list is a list of
words or phrases, and it might contain additional information, such
as example sentences for the word or the chapter the word appears in
the reference.

## Fields

The input file can have multiple fields for the target language. For
example, if the input file was created by export from a spreadsheet,
each field represents a column in the original spreadsheet.

## Tokens

We typically use an input notes file that is one row per English
word with a given meaning (or multiple English words with the same
meaning, but for simplicity, we describe the case with a single
English word). There may be multiple German words with this meaning,
in which case more than one German word may be entered in a single
German field. These multiple words would typically be separated by
commas or semi-colons as specified by the user. A token refers to
each piece of the field after splitting the field on the comma or
semi-colon delimiter.

## Headwords

The term ‘headword’ refers to the base form of a word that is
usually found in a dictionary. This is sometimes also called the
lemma. For example, in German the definite article indicates the
gender of the noun. The definite article and plural form are commonly
presented in lists of vocabulary, but neither is considered part of
the base form. Similarly, in German the infinitive form of the verb
is typically considered the base form. Therefore, when an reference
list of German words is provided, the headword is calculated by
removing the definite article or extracting the infinitive according
to some pre-determined rule (e.g., it might be known that for all
records representing verbs, the infinitive is the first verb form
listed). In the package, ‘headword’ refers to the keys for the
dictionaries containing audio file information or the dictionaries
representing reference lists. To avoid confusion, we call a string
a ‘lookup key’ if it will be used as a lookup in these two sets
of dictionaries, but if we are unsure if the lookup will be successful.

# Functionality

At a high level, the most important features are integrating audio
information into the deck, integrating information from multiple
reference lists into the deck (for example, to assign each note a
chapter for study), and creating tags useful for organizing studying.

Functionality applicable to most languages:

* Add audio files to deck and print the keys for the tokens where
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

The code below is an excerpt from examples/example1.py distributed
with the package. See the wiki for details on the structure of the
input notes file.

```python
# Dictionary for storing full path to each audio file, as well as the key
audio_file_dict = flawful.AudioFileDict()

# Load audio files to the dictionary
load_audios_to_dict(audio_file_dict, PRINT_DUPLICATE_AUDIO_HEADWORDS)
known_no_audio = make_known_no_audio_dict()

# Load reference lists into dictionaries
# reference lists added first take priority if a headword is in multiple lists
# (for example, when creating HTML tags for color highlighting)
de_dicts = flawful.Wordlists()
de_dicts.add(list_id='LA', chapter_offset=0, data=make_la_dict())
de_dicts.add(list_id='LB', chapter_offset=3, data=make_lb_dict())
de_dicts.add(list_id='LC', chapter_offset=5, data=make_lc_dict())

aud_dicts = {'file_info': audio_file_dict,
             'keys_no_audio': {},
             'known_no_audio': known_no_audio}

# Load okay-lists into dictionaries
de_okay_dicts = {}
de_okay_dicts['LA'] = make_okay_dict(os.path.join(INPUT_DIR,
                                     'okaylist_LA.txt'), 'LA')
de_okay_dicts['LB'] = make_okay_dict(os.path.join(INPUT_DIR,
                                     'okaylist_LB.txt'), 'LB')
de_okay_dicts['LC'] = make_okay_dict(os.path.join(INPUT_DIR,
                                     'okaylist_LC.txt'), 'LC')

#------------------------------------------------------------------------------
# Read input file that is one record per note for the flashcards.
#------------------------------------------------------------------------------
df = pd.read_csv(os.path.join(INPUT_DIR, 'input_notes.txt'), sep='\t',
                 skiprows=(0), na_filter=False, quoting=3)

flawful.dupkey(df, by_vars=['en1','part_of_speech'], desc='df',
               additional_vars=['input_note_id'], ifdup='error')
flawful.dupkey(df, by_vars=['input_note_id'], desc='df',
               additional_vars=['en1','de1'], ifdup='error')

df['note_id'] = df.input_note_id.map(lambda x: 'FLAWFUL_EX1_' + str(x))
df['de3'] = df.de3.map(flawful.german.show_vowel_length)
df['de_pronun'] = df.de_pronun.map(flawful.german.show_vowel_length)
df['de_conf'] = df.de_conf.map(preprocess_de_conf)
df['de1_sortable'] = df.de1.map(flawful.german.make_sortable_str)
res_mc = df.dech.apply(flawful.init_chapter,
                       str_to_chapter=str_to_chapter)
df['chapter'] = [ x['chapter'] for x in res_mc ]
df['chapter_tags'] = [ x['tags'] for x in res_mc ]

# This function does the most work.
# Field information passed in three or four equal-sized lists (`fields`,
#  `names`, `seps`, `assign_character`), but this may change in the future.
res_de = [
     flawful.tag_audio_and_markup(audio_dicts=aud_dicts, wordlists=de_dicts,
         str_to_wordlist_key=make_wordlist_key_notes,
         str_to_audio_key=make_audio_key_notes,
         select_keys_no_audio=filter_text_not_audio_pre,
         htag_prefix='DE',
         chapter=row[0],
         fields=[row[1],row[2],row[3],row[4],row[5],         row[6]],
         names= ['de1', 'at1', 'sd1', 'de3','dib_sentences','de_xref'],
         seps=  [','  , ','  , ','  , ';'  ,';'            ,';'      ],
         assign_chapter=[True, True, True, True, True, False],
         )
     for row in df[['chapter','de1','at1','sd1','de3','dib_sentences',
                    'de_xref']].values
         ]
# Put each element in `res_de` in own data frame column.
df['de_audio'] = [x.audio_output for x in res_de]
df['de_no_audio'] = np.where((df['de_audio'] == ''), 'no audio', '')
df['chapter'] = [x.chapter for x in res_de]
df['tags'] = [x.tags for x in res_de]
df['tags'] = df['chapter_tags'] + ' ' + df['tags']
for k in de_dicts.keys():
    df[f'In{k}'] = [x.in_wordlists[k] for x in res_de]
df['de1_color'] = [x.markup_output['de1'] for x in res_de]
df['de3_color'] = [x.markup_output['de3'] for x in res_de]
df['at1_color'] = [x.markup_output['at1'] for x in res_de]
df['sd1_color'] = [x.markup_output['sd1'] for x in res_de]
df['de_xref_color'] = [x.markup_output['de_xref'] for x in res_de]
df['de_sentences'] = [flawful.list_of_lists_to_str(x.sent_lists['LC'])
                 for x in res_de]
# done with `res_de`

# We use this as the primary answer for the flashcard, although we could also
# have simply left de1_color, at1_color, and sd1_color in separate
# fields on the back side of the flashcard.
df['de1_at1_sd1_color'] = np.select(
   [ (df.at1 != '') & (df.at1 == df.sd1),
     (df.at1 != '') & (df.sd1 != ''),
     (df.at1 != ''),
     (df.sd1 != '')],
   [ df.de1_color + '; A/CH: ' + df.at1_color,
     df.de1_color + '; A: ' + df.at1_color + '; CH: ' + df.sd1_color,
     df.de1_color + '; A: ' + df.at1_color,
     df.de1_color + '; CH: ' + df.sd1_color], default=df.de1_color)

df['de2_problems'] = [
         flawful.german.check_de2_problems(de1=row[0], de2=row[1],
                                           part_of_speech=row[2])
         for row in df[['de1','de2','part_of_speech']].values
                 ]

# Make a string with the number of words in the answer, e.g.
# '1/2 + A:1' means there are two answers in the de1 field (one is required
# and the other optional), and one answer in the at1 field.
df['de_target_number'] = [
          flawful.german.make_target_prompt(de1=row[0], sep=',',
                                            at1=row[1], sd1=row[2])
          for row in df[['de1','at1','sd1']].values
          ]
df['has_german_audio'] = df['de_audio'] != ''

#------------------------------------------------------------------------------
# Print words in various external lists that were not in the input notes
#------------------------------------------------------------------------------
de_dicts.print_unused_words(os.path.join(OUTPUT_DIR,
                                       'wordlists_headwords_not_in_notes.txt'),
                            WORDLISTS_TO_COMPARE, de_okay_dicts)

# Copy audio files to production directory and/or list unused files
if COPY_AUDIO: audio_file_dict.copy_used_files(AUDIO_OUTPUT_DIR)
if PRINT_UNUSED_AUDIO: audio_file_dict.print_unused_audio()

# Never sort when exporting all records, because the typical use case in
# that scenario is to create some new column that is then copied to the
# source spreadsheet. Otherwise, it's probably sensible to sort by chapter.
if not EXPORT_ALL_RECORDS:
    df = df.sort_values(['chapter','de1_sortable','en1','note_id'])

#dfout = df[(df.chapter <= MAX_CHAPTER)]
dfout = select_output_rows(df, MAX_CHAPTER, ONE_CHAPTER, EXPORT_ALL_RECORDS)

make_tables_and_listings(dfout, WORDLISTS_TO_COMPARE,
                         PRINT_NOTES_WITHOUT_AUDIO, ONE_CHAPTER)

flawful.german.write_de2_problems(dfout,
        os.path.join(OUTPUT_DIR, 'de2_problems.txt'))

if WRITE_WORDS_WITHOUT_AUDIO:
    flawful.german.write_keys_no_audio(
        os.path.join(OUTPUT_DIR, 'words_no_audio.txt'),
        aud_dicts['keys_no_audio'], filter_text_not_audio_post)

de_dicts.compare(dict_ids=WORDLISTS_TO_COMPARE)

#------------------------------------------------------------------------------
# Select fields for output and output
#------------------------------------------------------------------------------
dfout = select_output_columns(dfout)
dfout.to_csv(os.path.join(OUTPUT_DIR, f'{OUTPUT_FILE_PREFIX}.txt'), sep='\t',
             index=False, header=False, quoting=3)
# make an empty dataset and just write the column names
dfout[0:0].to_csv(os.path.join(OUTPUT_DIR, f'{OUTPUT_FILE_PREFIX}_fields.txt'),
                  sep='\t', index=False, quoting=3)

```

# Release Notes (v0.1.1)

* Update example1.py to copy audio files to output directory and upload the
  directory to the git repository (but do not include in the package)

* README.md changes

  - Add syntax highlighting in code example

  - Add links to git repository and wiki

# Other Resources

We have also authored the `wikwork` package that is designed to
retrieve word and audio file information from Wiktionary.
