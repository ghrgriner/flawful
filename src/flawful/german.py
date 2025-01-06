#    German-specific functions for language learning database
#    Copyright (C) 2020-2025 Ray Griner (rgriner_fwd@outlook.com)
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

"""German-specific functions for language learning database.

"""

#------------------------------------------------------------------------------
# File:   german.py
# Date:   2024-09-05
# Author: Ray Griner
# Purpose: German-specific functions for language learning database
#
# Changes:
# [20240905]: Created file by moving code here from create_dib.py
#------------------------------------------------------------------------------

import re
import csv
from typing import Optional, Dict, Callable
from .utils import OkaylistEntry

#------------------------------------------------------------------------------
# Regular expressions
#------------------------------------------------------------------------------
INITIAL_SICH_PATTERN = re.compile(r'^sich |^\(sich\) ')

# token has markup to show vowel length indicators, so it isn't a token where
# an audio file would be expected.
VOWEL_LENGTH_PATTERN = re.compile('<u>|ạ|Ạ|ẹ|Ẹ|ị|Ị|ọ|Ọ|ụ|Ụ|\u0323')

# All three definite articles and all 6 two-way and all 6 three-way
# combinations, followed by a space.
INITIAL_DER_DAS_DIE_PATTERN = re.compile('^der |^die |^das |^der/die '
 '|^der/das | ^das/der |^das/die |^die/der |^die/das |^der/die/das '
 '|^das/der/die |^der/das/die |^das/die/der |^die/der/das |^die/das/der ')

#-----------
# Functions
#-----------
def check_de2_problems(de1: str, de2: str, part_of_speech: str) -> str:
    """Create string indicating if secondary German field has issues.

    Parameters
    ----------
    de1 : str
        Primary German information. The general idea is that this will
        be nominative case for nouns, infinitive for verbs, separated by
        commas.
    de2 : str
        Secondary German information. This is expected to be plural
        endings for nouns and some conjugation information for verbs.
    part_of_speech : str
        'N' for noun, 'V' for verb. Any other value is ignored.

    Returns
    -------
    A string containing:
    - 'NOUN_TOKENS' if the number of tokens is wrong in the `de2` field
       for a noun. If `de1` contains '(in)' and has more than one token
       (using ',' as a separator), then `de2` tokenized by ';' should
       have the same number of tokens as `de1`. If `de1` contains '(in)'
       but has only one token, then `de2` should have two tokens
       (','-separated). If `de1` does not contain '(in)', then `de1` and
       `de2` should both be ','-separated with the same number of
       tokens.
    - 'VERB_TOKENS' if the number of tokens is wrong in the `de2` field
       for a verb.
    - 'VERB_PRONUN' if there is likely not sufficient information in the
       `de2` field to determine the verb's pronunciation (i.e., whether
       the verb is separable or inseparable). This is determined by
       checking if the headword from `de` starts with durch, um, unter,
       über, hinter, wider, or wieder, and if so, whether `de2`
       contains this same string.
    - '' otherwise.
    """
    de1 = de1.strip()
    de2 = de2.strip()
    def n_tokens(x: str, sep: str = ',') -> int:
        x = x.strip()
        if not x:
            return 0
        else:
            return (len(x.split(sep)))

    r_set = set()
    if part_of_speech == 'N':
        de1_list = de1.split(',')
        nexp = 0

        if '(in)' in de1:
            if ',' in de1:
                nexp = n_tokens(de1)
                nobs = n_tokens(de2, sep=';')
            else:
                nexp = 2
                nobs = n_tokens(de2)
        else:
            nexp = n_tokens(de1)
            nobs = n_tokens(de2)

        if not nexp == nobs:
            r_set.add('NOUN_TOKENS')

    if part_of_speech == 'V':
        nexp = n_tokens(de1)
        nobs = n_tokens(de2, sep=';')

        de1_list = de1.split(',')
        de2_list = de2.split(';')

        if nexp != nobs:
            r_set.add('VERB_TOKENS')
        else:
            for idx, token in enumerate(de1_list):
                headword = token.split('(')[0].split(' ')[-1]
                # TODO: rewrite using regex
                if (headword.startswith('durch')
                    and 'durch' not in de2_list[idx]):
                    r_set.add('VERB_PRONUN')
                if (headword.startswith('um')
                    and 'um' not in de2_list[idx]):
                    r_set.add('VERB_PRONUN')
                if (headword.startswith('unter')
                    and 'unter' not in de2_list[idx]):
                    r_set.add('VERB_PRONUN')
                if (headword.startswith('über')
                    and 'über' not in de2_list[idx]):
                    r_set.add('VERB_PRONUN')
                if (headword.startswith('hinter')
                    and 'hinter' not in de2_list[idx]):
                    r_set.add('VERB_PRONUN')
                if (headword.startswith('wider')
                    and 'wider' not in de2_list[idx]):
                    r_set.add('VERB_PRONUN')
                if (headword.startswith('wieder')
                    and 'wieder' not in de2_list[idx]):
                    r_set.add('VERB_PRONUN')

    ret_list = list(r_set)
    ret_list.sort()
    return ','.join(ret_list)

def show_vowel_length(x: str) -> str:
    """Convert bracketed vowel in input string to underline or underdot.

    Parameters
    ----------
    x : str
        Input text

    Returns
    -------
    A string where '[[' and ']]' are replaced with '<u>' and '</u>'
    repsectively (to show stressed long vowels by underlining), and
    '[v]' is replaced with a vowel with an underdot. For vowels without
    an umlaut diacritic, the pre-composed Unicode character is used.
    For vowels with an umlaut, U+0323 is put after the vowel.
    """
    ret_val = x.replace('[[', '<u>').replace(']]','</u>')
    ret_val = ret_val.replace('[a]', 'ạ').replace('[A]','Ạ')
    ret_val = ret_val.replace('[e]', 'ẹ').replace('[E]','Ẹ')
    ret_val = ret_val.replace('[i]', 'ị').replace('[I]','Ị')
    ret_val = ret_val.replace('[o]', 'ọ').replace('[O]','Ọ')
    ret_val = ret_val.replace('[u]', 'ụ').replace('[U]','Ụ')
    # Combining diacritics always go after the letter they modify, but
    # we have seen *nix terminals compose this wrong, even in 2025
    ret_val = ret_val.replace('[ü]', 'ü\u0323')
    ret_val = ret_val.replace('[Ü]', 'Ü\u0323')
    ret_val = ret_val.replace('[ä]', 'ä\u0323')
    ret_val = ret_val.replace('[Ä]', 'Ä\u0323')
    ret_val = ret_val.replace('[ö]', 'ö\u0323')
    ret_val = ret_val.replace('[Ö]', 'Ö\u0323')
    return ret_val

def make_sortable_str(x: str) -> str:
    """Convert a string for sorting (remove umlauts from vowels, etc.)

    Parameters
    ----------
    x : str
        Input text

    Returns
    -------
    A string where leading '(sich) ', 'sich ' or definite article(s) are
    removed, vowels have the umlaut diacritic removed, Eszett (ß) is
    converted to 'ss', and the string is made lower case. This might not
    be an exact implementation of some ISO-defined German sorting
    standard, but it's better than the default (when the locale isn't
    for Germany).
    """
    res1 = INITIAL_SICH_PATTERN.sub('', x)
    res2 = INITIAL_DER_DAS_DIE_PATTERN.sub('', res1).lower()
    res3 = res2.replace('ä','a').replace('ö','o').replace('ü','u')
    return res3.replace('ß','ss')

def make_target_prompt(de1: str, sep: str,
                       at1: Optional[str] = None,
                       sd1: Optional[str] = None,
                      ) -> str:
    """Make prompt giving number of tokens in answer.

    Parameters
    ----------
    de1 : str
        Input string with the words in the answer
    sep : str
        Separator for tokenizing `de1`, `at1`, and `sd1`.
    at1 : str, optional
        Austrian answer
    sd1 : str, optional
        Swiss German (schweizerdeutsch) answer

    Returns
    -------
    A string giving the number of tokens in the answer. The first number
    will represent the number of tokens in `de1`, which will typically
    be the German words/phrases commonly used in Germany (Deutschland)
    that will serve as the answer of the flashcard. If a token contains
    '°', the interpretation is that the token is for passive learning,
    and the learner need not be able to produce the answer to get the
    card correct. In this case, the string generated is 'N1/N2', where
    N1 are the number of tokens without '°' and N2 is the total.

    To this initial string is added the string '+ A:N3' if `at1` is
    populated (N3 is the number of tokens in `at1`) and '+ CH:N4' if
    `sd1` is populated. If `at1` and `sd1` are identical, then
    '+ A/CH:N3' is added instead.
    """
    de1_list = de1.split(sep)
    if at1 is not None:
        at1_list = at1.split(sep)
    else: at1_list = []
    if sd1 is not None:
        sd1_list = sd1.split(sep)
    else: sd1_list = []
    de1_tot_size = len(de1_list)
    de1_imp_size = len([val for val in de1_list if '°' not in val])

    if de1_imp_size == de1_tot_size:
        de_target = f'{de1_imp_size}'
    else:
        de_target = f'{de1_imp_size}/{de1_tot_size}'
    if at1 and at1 == sd1:
        if at1: de_target = f'{de_target} + A/CH:{len(at1_list)}'
    else:
        if at1: de_target = f'{de_target} + A:{len(at1_list)}'
        if sd1: de_target = f'{de_target} + CH:{len(sd1_list)}'
    return de_target

def write_keys_no_audio(outfile: str,
                        dict_: Dict[str, OkaylistEntry],
                        select: Callable[[str], bool]) -> None:
    """Write contents of dictionary to output file, if select(key) is True.
    """
    with open (outfile, 'w', encoding='utf-8') as csvfile:
        outwriter = csv.writer(csvfile, delimiter='\t',
                               quoting=csv.QUOTE_MINIMAL)
        outwriter.writerow(['Word'])
        for key in sorted(dict_.keys()):
            if select(key):
                outwriter.writerow([key])

