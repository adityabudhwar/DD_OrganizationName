import re
class Acronym:

    def __init__(self, phrase):
        self.phrase = self.bracesRemove(phrase)

    def bracesRemove(self, line):
        self.phrases_braces = []
        for m in re.finditer(r'(\(|\[).*?(\]|\))', line):
            self.phrases_braces.append(m.group().replace("(", "").replace(")", ""))
            line = line.replace(m.group(), '')
        return line

    def acronym1(self):
        prev = ''
        abbr = ''
        for c in self.phrase:
            if prev in " -" or prev.islower() and c.isupper():
                abbr += c.upper()
            prev = c
        return abbr

    def acronym2(self):
        return ''.join(w[0].upper() for w in self.phrase.split())

    def acronym3(self):
        return ''.join(w[0] for w in self.phrase.split() if w[0].isupper())

    stop = set(["a", "an", "the", "and", "nor", "yet", "so", "or", "both", "either", "neither", "only",
                "also", "whether", "when", "whenever", "where", "wherever", "while", "unless", "that",
                "though", "till", "if", "lest", "order", "even", "because", "although", "much", "soon",
                "aboard", "about", "above", "across", "after", "against", "along", "amid", "among", "anti",
                "around", "as", "at", "before", "behind", "below", "beneath", "beside", "besides", "between",
                "beyond", "but", "by", "concerning", "considering", "despite", "down", "during", "except",
                "excepting", "excluding", "following", "for", "from", "in", "inside", "into", "like", "minus",
                "near", "of", "off", "on", "onto", "opposite", "outside", "over", "past", "per", "plus",
                "regarding", "round", "save", "since", "than", "through", "to", "toward", "towards", "under",
                "underneath", "unlike", "until", "up", "upon", "versus", "via", "with", "within", "without"])

    def acronym4(self):
        return ''.join(w[0] for w in self.phrase.split() if (w[0].isupper() and w.lower() not in self.stop) )

    def acronym_braces(self):
        l = []
        for phrase in self.phrases_braces:
            if (len(phrase.split()) == 1):
                if(phrase.istitle):
                    l.append(phrase.lower())
        return  l

    def get_possible_acronyms(self):
        possible = set()
        possible.add(self.acronym1().lower())
        possible.add(self.acronym2().lower())
        possible.add(self.acronym3().lower())
        possible.add(self.acronym4().lower())
        possible.update(self.acronym_braces())
        list_p = []
        for x in possible:
           if (len(x) > 1):
              list_p.append(x)
        return list_p

if __name__ == '__main__':
    ac = Acronym("American Cancer Society Action Network (ACSANN)")
    print(ac.get_possible_acronyms())
