#    Miscellaneous utilities for language learning flashcards.
#    Copyright (C) 2024-2025 Ray Griner (rgriner_fwd@outlook.com)
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
#------------------------------------------------------------------------------

"""Miscellaneous utilities for language learning flashcards.

Functions that are not specific to any particular language are contained
here and have their names imported into the top-level namespace of the
package as well.
"""

#------------------------------------------------------------------------------
# File:    utils.py
# Date:    2024-12-05
# Author:  Ray Griner
# Purpose: Miscellaneous utilities
#------------------------------------------------------------------------------

import pandas as pd
import csv
from pathlib import Path
import shutil
from dataclasses import dataclass, field
from collections import namedtuple
from typing import Dict, Any, Callable, List, Union, Optional
import os

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------
DEFAULT_CHAPTER_FROM_WORDLIST = 9999
DEFAULT_CHAPTER_FROM_NOTES = 999

#------------------------------------------------------------------------------
# Class definitions
#------------------------------------------------------------------------------
# Entry in the dictionary that will store audio file information.
#   path: path to the file locations
#   in_deck: Boolean indicator whether the audio file correponds to a word in
#     the source wookbook
#------------------------------------------------------------------------------
class AudioFileDictEntry:
    """Class for audio file information.

    Attributes
    ----------
    path : pathlib.Path
        Path to file
    in_deck : bool, default = False
        Indicator whether the file is used anywhere in the flashcard
        deck
    basename_out : str
        Output file basename (excluding the final '.' and extension)
    ext : str
        File name extension (e.g., 'mp3')
    """
    def __init__(self, path: Path, basename_out: str, ext: str):
        self.path = path
        self.in_deck = False
        self.basename_out = basename_out
        self.ext = ext
    def __str__(self) -> str:
        return (f'({self.path=},{self.in_deck=},{self.basename_out=},'
                '{self.ext=})')
    def __repr__(self) -> str:
        return str(self)

    def audio_markup(self) -> str:
        """Return string with Anki markup to play the file.
        """
        return f'[sound:{self.basename_out}.{self.ext}]'

@dataclass
class AudioFileNameInfo:
    """Simple class containing the headword and output name for an audio file.

    Attributes
    ----------
    headword : str
        Headword associated with the audio file.
    basename_out : str
        Basename of the output file (excluding file extensions like
        '.mp3' or '.ogg'.
    """
    headword : str
    basename_out : str

class AudioFileDict(dict):
    """Dictionary storing audio file information for each headword.

    The values in the dictionary are AudioFileDictEntry objects.
    """
    def add_from_dir(self,
                     dirpath: str,
                     ext: str,
                     make_name_info: Callable[[str], AudioFileNameInfo],
                     recurse: bool = False,
                     print_duplicate_headword: bool = True
                    ) -> None:
        """Add files to the dictionary from a directory and/or subdirectories.

        Parameters
        ----------
        dirpath : str
            String giving the path to a directory that will be searched.
        ext : str
            File extension of input files. This will be used when
            calling Path.rglob, so the usual file name globbing rules
            apply.
        make_name_info : str
            A function that takes the path stem (str) and returns an
            AudioFileNameInfo object (with the headword and file
            basename).
        recurse : bool, default = False
            Recursively search subdirectories under `dirpath` as well.
        print_duplicate_headword : bool, default = True
            Controls whether a warning message is printed to stdout if
            the headword already exists in the dictionary.

        Returns
        -------
        None
        """
        if recurse:
            path_iter = Path(dirpath).rglob(f'*.{ext}')
        else:
            path_iter = Path(dirpath).glob(f'*.{ext}')

        if ext is None: ext = '*'
        for path in path_iter:
            name_info = make_name_info(path.stem)
            dictkey = name_info.headword
            if (print_duplicate_headword and (dictkey in self)):
                print(f'Duplicate headword in audio: {dictkey} (current: '
                      f'{path.resolve()}, previous: {self[dictkey].path})')
            else:
                self[dictkey] = AudioFileDictEntry(path.resolve(),
                                                   name_info.basename_out,
                                                   ext)

    def copy_used_files(self, outdir: str) -> None:
        """Copy files used in the deck (in_deck == True) to a new directory.

        If the deck is subset after the `in_deck` values were
        set, some audio files might be unnecessarily copied to the
        output. In this case, (1) the unneeded files can be removed by
        the Anki check media function (if the new directory is the Anki
        media directory), or (2) alternatively, the user can call
        `tag_audio_and_markup` once to calculate the chapter for each
        note, subset the data frame, call `reset_in_deck` and then run
        `tag_audio_and_markup` again.
        """
        for _, value in self.items():
            if value.in_deck:
                shutil.copyfile(value.path,
                    os.path.join(outdir, f'{value.basename_out}.{value.ext}'))

    def reset_in_deck(self) -> None:
        """Reset in_deck attribute of all values to False.

        See `copy_used_files` above for rationale.
        """
        for _, v in self.items():
            v.in_deck = False

    def print_unused_audio(self) -> None:
        """Print path to stdout for all files where in_deck == False.
        """
        any_audio_not_in_input = False
        for _, value in self.items():
            if not value.in_deck:
                if not any_audio_not_in_input:
                    print('\nUnused audio files:')
                    any_audio_not_in_input = True
                print(value.path)

@dataclass
class WordlistEntry():
    """ Entry in word list.

    Attributes
    ----------
    full_line : str
        The line from which the entry was extracted. If the full line is
        not readily available (e.g., if the entry is from a data frame
        that was already processed), then give the field with the most
        information containing the headword.
    examples : list[str], optional
        Example sentences or expressions using the word.
    study_chapter : int (default: DEFAULT_CHAPTER_FROM_WORDLIST)
        Give the chapter the word will be studied. Every note in a
        flashcard deck is assigned a chapter, and this field is the
        minimum across the notes. This could be less than the book
        chapter (since a note could contain words from earlierchapters)
        or greater than the book chapter (e.g., if the word is not
        included on any notes).
    book_chapter: int (default: DEFAULT_CHAPTER_FROM_WORDLIST)
        If the word is from a book or some other resource where a
        chapter is assigned, this is that chapter.
    """
    full_line: str = ''
    examples: List[str] = field(default_factory=list)
    study_chapter: int = DEFAULT_CHAPTER_FROM_WORDLIST
    book_chapter: int = DEFAULT_CHAPTER_FROM_WORDLIST

@dataclass
class OkaylistEntry():
    """ Entry in okay-list.

    Attributes
    ----------
    full_line : str
        The line from which the entry was extracted.
    """
    full_line: str = ''

class Wordlist:
    """Word list words and meta-data

    Attributes
    ----------
    data : dict
        Dictionary that takes a headword (str) as key and has values
        that is a WordlistEntry object
    tag : str
        If a word is in the word list, this tag will be created for the
        note.
    html_tag :
        If a word is in the wordlist, the output will have and HTML font
        class added with this tag.
    chapter_offset : int, optional (default: 0)
        Value that should be added to the data[some_key].book_chapter
        value to calculate the chapter associated with each word. The
        idea is that different Wordlists should not return the same
        calculated chapter, because the Wordlists represent the order of
        books or other resource the user is studying.
    pos : int (default: DEFAULT_CHAPTER_FROM_WORDLIST)
        The position of the object in the Wordlists() object. 0 is the
        first inserted, 1 is the second inserted, etc...
    force_chapter : int, optional (default: None)
        If populated, the calculated chapter for any word in the list
        should be set to force_chapter.
    """
    def __init__(self,
                 data: Dict[str, WordlistEntry],
                 tag: str,
                 html_tag: str,
                 chapter_offset: Optional[int] = 0,
                 pos: int = DEFAULT_CHAPTER_FROM_WORDLIST,
                 force_chapter: Optional[int] = None):
        if chapter_offset is None:
            self.chapter_offset = 0
        else:
            self.chapter_offset = chapter_offset
        self.force_chapter = force_chapter
        self.data = data
        self.tag = tag
        self.html_tag = html_tag
        self.pos = pos

class Wordlists(dict):
    """Dictionary where each entry is a different word list.
    """
    def add(self,
            list_id: str,
            data: Dict[str, WordlistEntry],
            tag: Optional[str] = None,
            html_tag: Optional[str] = None,
            chapter_offset: Optional[int] = None,
            force_chapter: Optional[int] = None
           ) -> None:
        if list_id in self:
            raise ValueError(f'{list_id=} already exists')
        if tag is None: tag = list_id
        if html_tag is None: html_tag = list_id
        self[list_id] = Wordlist(chapter_offset=chapter_offset, tag=tag,
                pos=len(self), data=data, html_tag=html_tag,
                force_chapter=force_chapter)

    def print_unused_words(self,
                outfile: str,
                dict_ids: Union[str, List[str]],
                okay_dicts: Dict[str, Dict[str, OkaylistEntry]]) -> None:
        """Print words from word list(s) that were not used in the deck.

        Parameters
        ----------
        outfile : str or other object that can be passed to open()
        dict_ids : str or list[str]
            List of keys identifying the word lists to print, or 'All'
        okay_dicts : dict
            Dictionary where each value is a dictionary whose records
            are OkaylistEntry. An unused word is not printed if it is
            in the okay-list with the same key as the Wordlist.
        """
        if dict_ids == 'All':
            cefr_list = self.keys()
        else:
            cefr_list = dict_ids

        if cefr_list: print()
        with open(outfile, 'w', encoding='utf-8') as csvfile:
            outwriter = csv.writer(csvfile, delimiter='\t',
                                   quoting=csv.QUOTE_MINIMAL)

            for dict_id in cefr_list:
                not_found=0
                found_or_ok=0
                found_in_db=0
                okay_dict = okay_dicts.get(dict_id, {})
                lookup_dict = self[dict_id].data
                for key, value in lookup_dict.items():
                    if (value.study_chapter == DEFAULT_CHAPTER_FROM_WORDLIST
                           and not key in okay_dict):
                        outwriter.writerow([f'Not in DB: {dict_id}: {key}',
                            value.full_line, value.book_chapter])
                        not_found = not_found + 1
                    elif (value.study_chapter == DEFAULT_CHAPTER_FROM_NOTES
                           and not key in okay_dict):
                        outwriter.writerow([f'No chapter assigned: {dict_id}: '
                            f'{key}', value.full_line, value.book_chapter])
                        found_in_db = found_in_db + 1
                    else:
                        found_or_ok = found_or_ok + 1
                print(f'List {dict_id}: chapter assigned or in okay-list '
                      + f'{found_or_ok}, in DB but no chapter assigned '
                      + f'{found_in_db}, not in DB {not_found}')

    def compare(self,
                dict_ids: Union[str, Union[str, List[str]]] = 'All') -> None:
        """Print one-way frequency tables and pivot table giving counts.

        So if len(dict_ids) == 3, a pivot table is printed giving counts
        for the 8 possible combinations (Y/N) x (Y/N) x (Y/N) for the
        three lists.

        Parameters
        ----------
        dict_ids : str or list[str], optional (default = 'All')
            List of keys identifying the word lists to print. If 'All',
            then all keys are used.
        """
        if dict_ids is None or dict_ids == 'All':
            dict_ids = self.keys()
        for idn, idval in enumerate(dict_ids):
            if idn == 0:
                cum_df = pd.DataFrame(self[idval].data.keys(),
                                      columns=['headword'])
                cum_df['In_' + idval] = 'Y'
                cum_df.set_index('headword')
            else:
                new_df = pd.DataFrame(self[idval].data.keys(),
                                      columns=['headword'])
                new_df['In_' + idval] = 'Y'
                new_df.set_index('headword')
                cum_df = cum_df.merge(new_df, how='outer', on='headword')
        cum_df.fillna('N', inplace=True)

        print()
        for idval in dict_ids:
            print(f'{idval}: {len(self[idval].data)} unique headwords (=keys)')
        addin = ['In_' + x for x in dict_ids]
        print('\nCross-tabulated frequency counts by reference list')
        print(cum_df.pivot_table(index=addin, values='headword',
              aggfunc=len, margins=True))

@dataclass
class AudioMarkupRec:
    """Class for markup and audio info output from `tag_audio_and_markup`.

    See that function's documentation for details.
    """
    audio_output: Union[str, Dict[str, str]]
    chapter: int
    tags: str
    in_wordlists: Dict[str, bool]
    markup_output: Dict[str, str]
    sent_lists: Dict[str, List[str]]

_TokenInfo = namedtuple('_TokenInfo', 'index value')
#------------------------------------------------------------
# End class definitions
#------------------------------------------------------------
#------------------------------------------------------------------------------
# Function definitions
#------------------------------------------------------------------------------
# TODO: remove Any from audio_dicts
# TODO: consider changing interface to something other than passing four
#   equal-sized lists
def tag_audio_and_markup(audio_dicts: Dict[str, Any],
        wordlists: Wordlists,
        str_to_wordlist_key: Callable[[str], str],
        str_to_audio_key: Callable[[str], str],
        select_keys_no_audio: Callable[[int, Dict[str, bool], str], bool],
        htag_prefix: str,
        chapter: int,
        fields: List[str],
        names: List[str],
        seps: List[str],
        assign_chapter: Optional[List[bool]] = None,
        first_chapter_tag_prefix: str = 'FC',
        in_wordlists_tag_prefix: str = 'In',
        audio_output_format: str = 'str',
        ) -> AudioMarkupRec:
    """Create output with audio and markup info for input fields.

    This is the most important function in the package. It takes as
    input equal-sized lists of fields, field names, and separators, that
    contain the words or phrases the users wants to study (other
    material such as English comments can also be mixed in). The program
    parses each field to attempt to extract the headword for each token.
    That is, it parses each token to make lookup keys for the word lists
    and audio file dictionaries. (For the sake of clarity, we refer to
    this lookup key as a headword only if it was found in a dictionary.)
    Then:

    (1) the function checks if an audio file exists for the lookup key,
    and if so, it generates the Anki syntax to play the audio file. It
    also marks the audio file as found for a subsequent function (if
    called) to copy the file to the directory required by Anki.

    (2) the function checks if the lookup key exists in one of the
    user-provided word lists, and if so, it

    (a) adds HTML markup surrounding the token to the field so that the
        word will be highlighted in the flashcards (e.g., by making all
        words in some CEFR A1 list blue, or bold, etc...).
    (b) gets the example sentences for the headword from the Wordlist
        and makes an HTML unordered list of the sentences.
    (c) gets the chapter information, if it exists, from the Wordlist,
        and updates the chapter for the note if necessary to obtain the
        minimum of the `chapter` parameter and the chapters found for
        all the headwords.

        If this chapter is less than the `study_chapter` for the
        headword in the Wordlist, then the `study_chapter` is set to
        the chapter for the note.

    Parameters
    ----------
    audio_dicts : dict
        Dictionary containing three other dictionaries.
        file_info : path to audio file for a given lookup key (headword)
        keys_no_audio : used to store tokens that yielded lookup keys
          for which no audio was found.
        known_no_audio : lookup keys for which no audio is expected
          (e.g., because user has already checked for audio)
    wordlists : Wordlists object
        This is a dictionary where the key is a tag defined by the user
        and each entry is a Wordlist object.
    str_to_wordlist_key: function
        User-defined function that takes a token (str) as input and
        returns the key for lookup in the `Wordlist` objects. For
        example, the token might be 'das Boot' (the boat), and the
        returned key is 'Boot'.
    str_to_audio_key : function
        Like str_to_wordlist_key, but it returns a string that generated
        in the same way the keys to audio_dicts['file_info'] were parsed
        (see above).  In other words, the function will return a string
        that will be used to attempt a look-up in
        audio_dicts['file_info']
    select_keys_no_audio : function
        Function that takes as input a chapter, a dictionary that
        indicates whether the key was found in each Wordlist in
        `dicts`, and the token used to generate the key. This
        function should return a boolean indicating whether the token
        should be added to `audio_dicts['keys_no_audio']` if no audio
        file was found.
    htag_prefix : string
        If the lookup key matches one of the wordlists, the token will
        be tagged with an HTML '<font class={htag_prefix}{htag}>', where
        htag is taken from the `tag` attribute of the matched Wordlist
        object.
    chapter : int
        Minimum chapter found so far for the note.
    field_values : list[str]
        Data in the fields that the function will parse to create the
        lookup keys.
    field_names : list[str], same length as `field_values`
        Names of the fields in the `field_values` parameter. These
        names will be passed to the function output. The current
        interface is passing 3 or 4 equal sized lists (`field_values`,
        `field_names`, `seps`, `assign_chapter`) but this may change in
        the future.
    seps : list[str], same length as `field_values`
        List of separators that gives the character used to separate the
        tokens in each input field.
    assign_chapter : list[bool], optional (default = None)
        List of indicators whether the field should be used to assign
        the chapter. The purpose is to allow fields that are only used
        to link to audio files. If not None, should be same length as
        `field_values`.
    first_chapter_tag_prefix : str, optional (default = 'FC')
        The final chapter will be calculated for the note (by extracting
        each headword, seeing if there is a chapter assigned if a 
        Wordlist, and taking the minimum of these and the `chapter`
        value). Once the minimum chapter is calculated, a tag will be
        added to the `Tags` output string. This gives the prefix of
        that tag.
    in_wordlists_tag_prefix : str, optional (default = 'In')
        If a headword is found in a Wordlist (see `wordlists`
        parameter), then a tag will be created
        '{in_wordlists_tag_prefix}{tag}', where tag is the attribute
        from the matched Wordlist object.
    audio_output_format : str, {'str' (default) or 'dict'}
        Determines whether `markup_output` is a single string, or a
        dictionary of strings with keys matching the values passed to
        the `names` input parameter.

    Returns
    -------
    An `AudioMarkupRec` object with the following attributes:
      audio_output : str or dict
        For any headwords where audio files were found, the Anki syntax
        is generated to play the matching audio file in the order found,
        e.g., '[sound:eye.mp3][sound:eyelash.mp3]'. This is either a
        single string for the audio files found for all the input
        `fields`, or a dictionary where the keys are the field names
        (from the `names` parameter) and the value is the audio
        associated with that field.
      chapter : int
        Minimum chapter calculated (see `first_chapter_tag_prefix`
        parameter above for details).
      tags : str
        Single string containings the tags generated, as described in
        the `first_chapter_tag_prefix` and `in_wordlists_tag_prefix`
        parameter descriptions.
      in_wordlists : str
        A dictionary with keys matching `dicts.keys()` where the value
        is whether a lookup key from any token of any parsed field was
        found matching the given `Wordlist` in `dicts`.
      markup_output : dict
        A dictionary with keys matching `names` where the value is the
        corresponding item in `fields` with markup (possibly) added as
        described in the `htag_prefix` parameter description above.
      sent_lists : list[str]
        A dictionary with keys matching `wordlists.keys()` where the
        value is a HTML string with the example sentences from the
        matching Wordlist object, if found. If a headword is repeated,
        the sentences are also repeated. The HTML string presents the
        sentences as an unordered list.
    """
    if assign_chapter is None:
        assign_chapter = [ True for _ in range(len(fields)) ]
    # tokens from all the fields in one list
    tokens = []
    for idx, val in enumerate(fields):
        field_list = val.split(seps[idx])
        tokens.extend([_TokenInfo(index=idx, value=val)
                       for val in field_list])
    audio_output: Union[str, Dict[str, str]]
    if audio_output_format == 'str':
        audio_output = ''
    elif audio_output_format == 'dict':
        audio_output = {}
    else:
        raise ValueError('{audio_output_format=} must be str or dict')

    # Indicate whether >=1 of the words in the record is in the given
    # word list.
    in_wordlists: Dict[str, bool] = {}
    sent_lists: Dict[str, List[str]] = {}
    for key in wordlists.keys():
        in_wordlists[key] = False
        sent_lists[key] = []
    #out_list = [ [], [], [], [], [], [] ]
    # [ [] ] * len(fields) won't work, they all refer to the same list!
    out_list: List[List[str]] = [ [] for _ in range(len(fields)) ]

    # Loop 1: Get lowest CEFR level of any word presented in the note
    for token in tokens:
        max_word_level = len(wordlists.keys())
        word_level = max_word_level
        wordlist_key = str_to_wordlist_key(token.value)
        audio_key = str_to_audio_key(token.value)

        for dict_id in wordlists.keys():
            if wordlist_key in wordlists[dict_id].data:
                in_wordlists[dict_id] = True
                sent_lists[dict_id].append(
                        wordlists[dict_id].data[wordlist_key].examples)
                word_level = min(word_level, wordlists[dict_id].pos)
                force_chapter = wordlists[dict_id].force_chapter
                if force_chapter is None and assign_chapter[token.index]:
                    chapter = min(chapter,
    (wordlists[dict_id].chapter_offset
     + wordlists[dict_id].data[wordlist_key].book_chapter))
                elif assign_chapter[token.index]:
                    chapter = min(chapter,
                                      wordlists[dict_id].force_chapter)

        if word_level < max_word_level:
            # TODO: efficiency?
            htag = wordlists[list(wordlists.keys())[word_level]].html_tag
            out_list[token.index].append(
                 f'<font class={htag_prefix}{htag}>{token.value}</font>')
        elif word_level == max_word_level:
            out_list[token.index].append(token.value)
        else:
            raise ValueError(f'Invalid {word_level=}!')

    markup_output = {}
    for idx, val in enumerate(out_list):
        markup_output[names[idx]] = seps[idx].join(val)

    # Loop 2: Get audio for each word, and print out words where the audio is
    # missing. This is its own loop so we can filter by properties of the
    # whole note. This is so that if we want to study notes assigned to A1,
    # it seems better to get audios for all words on the note (some might be
    # A2 or B1) instead of just getting audios for the A1 words.
    for token in tokens:
        wordlist_key = str_to_wordlist_key(token.value)
        audio_key = str_to_audio_key(token.value)

        # Update again with the chapter
        for dict_id in wordlists.keys():
            wordlist_entry = wordlists[dict_id].data.get(wordlist_key, None)
            if wordlist_entry is not None:
                if chapter < wordlist_entry.study_chapter:
                    wordlist_entry.study_chapter = chapter

        # Check for a match with audio files.
        audio_found = False

        if audio_key:
            aud_file_val = audio_dicts['file_info'].get(audio_key)
            if aud_file_val:
                if audio_output_format == 'str':
                    audio_output += (
                        aud_file_val.audio_markup())
                elif audio_output_format == 'dict':
                    audio_output[names[token.index]] += (
                        aud_file_val.audio_markup())
                aud_file_val.in_deck = True
                audio_found = True

        if ((not audio_found) and audio_key
             and select_keys_no_audio(chapter=chapter,
                                      in_wordlists=in_wordlists,
                                      token=token.value)):
            if not audio_key in audio_dicts['known_no_audio']:
                audio_dicts['keys_no_audio'][audio_key] = token.value

    tags=f'{first_chapter_tag_prefix}{chapter}'

    tagset=set()
    for k, v in in_wordlists.items():
        if v:
            tag = wordlists[k].tag
            if tag not in tagset:
                tags = tags + f' {in_wordlists_tag_prefix}{tag}'
                tagset.add(tag)

    return AudioMarkupRec(audio_output=audio_output, chapter=chapter,
             tags=tags, in_wordlists=in_wordlists,
             markup_output=markup_output, sent_lists=sent_lists)

def dupkey(df: pd.DataFrame,
           by_vars: List[str],
           desc: str ='dataset',
           additional_vars: Optional[List[str]] = None,
           ifdup: str = 'error',
           print_duplicates: bool = True,
           where: Optional[pd.Series] = None
           ) -> bool:
    """Check if data frame has duplicates and print them, raise error, etc.

    Throws a ValueError() if ifdup='error' and duplicate found.

    Parameters
    ----------
    df : pandas.DataFrame
    by_vars : list[str]
        List of columns in the data frame to use as key when checking
        for duplicates.
    desc : str
        Description of the data frame to be printed with the count of
        the duplicates.
    additional_vars : list[str], optional
        Additional variables in the data frame to be printed with the
        variables in `by_vars`.
    ifdup : {'warning', 'error'}
        Behavior if duplicate(s) are found. If 'error', then ValueError
        is raised.
    print_duplicates : bool, default = True
        Controls whether duplicate records are printed. If False, then
        only the summary message is printed.
    where : pandas.Series
        Boolean series or similar-object for selecting rows of the data
        frame before checking for duplicates.

    Returns
    -------
    True if duplicate(s) found, and False otherwise.
    """
    if ifdup not in ('error','print'):
        raise ValueError("ifdup must equal 'error' or 'print'")
    if additional_vars:
        collist = by_vars + additional_vars
    else:
        collist = by_vars
    if not where:
        df2 = df[collist].sort_values(by_vars)
    else:
        df2 = df[where][collist].sort_values(by_vars)
    duprows = df2.duplicated(subset=by_vars, keep=False)
    if duprows.any():
        print(f'\ndupkey: {desc} by {by_vars} has {sum(duprows)} rows where'
              ' key is a duplicate\n')
        if print_duplicates:
            print(df2[duprows])
        if ifdup == 'error':
            raise ValueError(f'Duplicate value in {by_vars}')
        return True
    else:
        return False

def list_of_lists_to_str(input_list: List[List[str]],
                          fmt: str = 'ulhtml'
                         ) -> str:
    """Convert a list of lists to a string formatted as HTML.

    Parameters
    ----------
    input_list : list[list[str]]
        Input list of list[str]. Input_list contains all the example
        sentences for all the headwords for a given note.
    fmt : str (default = 'ulhtml')
        Output format taking one of three values
        'ulhtml' - Inner list written as  string containing unordered
                   list ('<ul>...</ul>'). Output is then concatenated.
        'olhtml' - Inner list written as  string containing ordered list
                   ('<ol>...</ol>'). Output is then concatenated.
        'brhtml' - Inner list separates items only by '<br>'. Outer list
                   concatenates, adding another '<br>' between lists.
    """

    out_list = []
    for item in input_list:
        if fmt == 'ulhtml':
            out_str = '</li><li>'.join(item)
            if out_str:
                out_str = f'<ul><li>{out_str}</li></ul>'
                out_list.append(out_str)
        elif fmt == 'olhtml':
            out_str = '</li><li>'.join(item)
            if out_str:
                out_str = f'<ol><li>{out_str}</li></ol>'
                out_list.append(out_str)
        elif fmt == 'brhtml':
            out_str = '<br>'.join(item)
            if out_str:
                out_list.append(out_str)

    if not out_list:
        final_out = ''
    elif fmt != 'brhtml':
        final_out = ''.join(out_list)
    else:
        final_out = '<br>'.join(out_list)
    return final_out

def twowaytbl(df: pd.DataFrame,
              row: Union[str,List[str]],
              col: Union[str,List[str]],
              title: str = '',
              cumulative: bool = False
             ) -> None:
    """Wrapper around pd.crosstab to add title and optional cumulative column.

    Parameters
    ----------
    df : pd.DataFrame
        Input data frame
    row : str or list[str]
        Column(s) from df that will give the rows in the cross-
        tabulation.
    col : str or list[str]
        Column(s) from df that will give the columns in the
        cross-tabulation.
    title : str, default = ''
        Title to print
    cumulative : bool, default = False
        Add cumulative sum column to cross-tabulation.
    """
    if title:
        print(title)
    if df.empty:
        print('Empty data frame')
        return
    ctab = pd.crosstab(index=df[row], columns=df[col], margins=True)
    if cumulative:
        ctab_nomarg = pd.crosstab(index=df[row], columns=df[col])
        ctab_mrg = pd.merge(ctab, ctab_nomarg.cumsum(), on=[row],
            how='outer', suffixes = ['','_cum'])
        print(ctab_mrg)
    else:
        print(ctab)

def init_chapter(chapters: str,
                     str_to_chapter: Callable[[str], int],
                     tag_prefix: str = 'Ch',
                     sep: str = ','
                    ) -> Dict[str, Union[int,str]]:
    """Given a str that is a delimited list of chapters, get the minimum.

    The values in the delimited list need not be numbers. They could,
    for example, be B1-B10 (representing chapters 1-10 in book/resource
    B), and these strings are then converted to a number by
    `str_to_chapter`. A 'tags' string is also created with a tag for
    each chapter in the list (after calling str_to_chapter())

    Parameters
    ----------
    chapters : str
        Delimited list of chapters
    str_to_chapter: function
        Function that takes a string as input and returns the chapter
        (int)
    tag_prefix : str (default='Ch')
        Prefix added before the chapter in the created tag(s)
    sep : str (default=',')
        Delimiter of chapters.

    Returns
    -------
    A dictionary with two items, `chapter` and `tags`.
    """
    chaplist=[str_to_chapter(x) for x in chapters.split(sep)]
    chapter = min(chaplist)
    chaplist = sorted(list(set(chaplist))) # remove duplicates
    taglist = [f'{tag_prefix}{x}' for x in sorted(chaplist)]
    tags = ' '.join(taglist)
    return {'chapter': chapter, 'tags': tags}

