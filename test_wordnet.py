#!/usr/bin/env python3
"""Test WordNet installation and count synsets."""

from nltk.corpus import wordnet as wn

all_synsets = list(wn.all_synsets())
print(f"Total synsets in WordNet: {len(all_synsets)}")
print(f"\nSample synsets:")
for i, ss in enumerate(all_synsets[:10]):
    print(f"  {i+1}. {ss.name()}: {ss.definition()}")

print(f"\nVerbs: {len(list(wn.all_synsets('v')))}")
print(f"Nouns: {len(list(wn.all_synsets('n')))}")
print(f"Adjectives: {len(list(wn.all_synsets('a')))}")
print(f"Adverbs: {len(list(wn.all_synsets('r')))}")
