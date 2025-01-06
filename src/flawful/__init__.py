#   Managing language learning flashcards, audio files, and word lists.
#   Copyright (C) 2020-2025 Ray Griner (rgriner_fwd@outlook.com)
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

"""Managing language learning flashcards, word lists, and audio files.

Summary
-------

This package has functionality for maintaining and managing language
learning flashcards and associated audio files.

The primary objective of sharing this package and documentation is to
aid users who ask the question “How can I use spaced-repetition
flashcard software to learn a foreign language?” We present
suggestions with example code for the case of an English native-speaker
studying German, but the code is designed for use with arbitrary
languages (although users might need to write their own functions
that can be input to the package functions to parse their input).

Functionality
-------------

At a high level, the most important features are integrating audio
information into a deck, integrating information from multiple reference
lists into a deck (for example, to assign each note a chapter for
study), and creating tags useful for organizing studying.

Example
-------

An example using about 40 German notes is distributed with the package
using vocabulary and audio files taken from the open-souce textbook
Deutsch im Blick and its online companion grammar Grimm Grammar. (See
LICENSE.txt for licensing details.)
"""

#------------------------------------------------------------------------------
# File:    __init__.py
# Date:    2024-12-19
# Author:  Ray Griner
#
#------------------------------------------------------------------------------
__author__ = 'Ray Griner'
__version__ = '0.1.1'
__all__ = ['german','utils']

#from .german import write_tokens_wo_audio, make_de2_probs
#from .german import show_vowel_length, make_tables_and_listings
#from .german import write_de2_prob, make_target_prompt, make_sortable_str
#from .german import VOWEL_LENGTH_PATTERN, INITIAL_SICH_PATTERN
#from .german import INITIAL_DER_DAS_DIE_PATTERN
from .utils import AudioFileDict, dupkey, init_chapter, twowaytbl
from .utils import list_of_lists_to_str, Wordlists, tag_audio_and_markup
from .utils import AudioFileNameInfo, WordlistEntry, OkaylistEntry
from .utils import DEFAULT_CHAPTER_FROM_NOTES

