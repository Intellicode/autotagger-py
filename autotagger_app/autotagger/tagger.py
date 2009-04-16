from autotagger.stop_words import STOPWORDS
from autotagger.stemmer import PorterStemmer
import datetime 
from whitelist import WHITELIST
from constants import TAG_CONSTANTS


import re
"""
       A U T O T A G S
       Automatic tag suggestions or keyword generation for text, using unsupervised
       semantic analysis.

       Copyright (C) 2007  Hjortur Stefan Olafsson

       This program is free software: you can redistribute it and/or modify
       it under the terms of the GNU General Public License as published by
       the Free Software Foundation, either version 3 of the License, or
       (at your option) any later version.

       This program is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
       GNU General Public License for more details.

       You should have received a copy of the GNU General Public License
       along with this program.  If not, see <http://www.gnu.org/licenses/>.

       @version 1.2

       TODO Remove redundant lowercasing
       TODO Choose best inflection after stemming (based on frequency)
       TODO Apply further weighting based on position, applying more weighting for terms that appear at the beginning
       TODO Separate out language specific regular expressions
       TODO Support multiple sets of text with different weightings (field weights)
       TODO Add support for "associated" tags (e.g when suggesting 'lucene' also add 'search', 'java')
       TODO Support term normalisation ( 'youtube', 'iFilm' -> 'video sharing' )

"""

AUTOTAGS = {
        'NAME' : 'AutoTags',
        'VERSION' : 1.2,
        'DEFAULT_COMPOUND_TAG_SEPARATOR' : ' ',
        'APPLY_STEMMING' : True, # If true then the Porter stemmer should be applied to all tokens (but not phrases or n-grams), this has some overhead
        'BOUNDARY' : '##!##' # Compound terms will not be created across BOUNDARIES
}

TermConstants = {
        'TYPE_SINGLE_TERM' : 'TYPE_SINGLE_TERM',
        'TYPE_CAPITALISED_COMPOUND_TERM' : 'TYPE_CAPITALISED_COMPOUND_TERM',
        'TYPE_SIMPLE_BIGRAM_TERM' : 'TYPE_SIMPLE_BIGRAM_TERM',
        'TYPE_SPECIAL_TERM' : 'TYPE_SPECIAL_TERM',
        'TYPE_TAG_CONSTANT' : 'TYPE_TAG_CONSTANT'
}

class Term:
    def __init__(self):
        self._termId = ''
        self._term = ''
        self.termType = TermConstants['TYPE_SINGLE_TERM']
        self.freq = 1
        self.ignoreTermFreqCutoff = False
        self.score = 0
        self.boost = 1

    def setTermType(self,type):
        self.termType = type
    def addBoost (self, boostFactor ):
        self.boost *= boostFactor
    
    def setBoost(self,boost):
        self.boost = boost

    def incrementFrequency(self):
        self.freq += 1
       
    def getScore(self):
        self.score = self.freq*self.boost;
        return self.score
       
    def getValue(self):
        return self._term
       
    def setValue (self,value ):
        self._term = value
        self._setTermId()
       
    def getTermId(self):
        return self._termId

       
    def _setTermId(self):
        # If this is a single token and stemming should be applied then modify the termID
        if AUTOTAGS['APPLY_STEMMING'] and not self.isCompoundTerm():
            self._termId = _stemToken( self.getValue() )
        else:
            self._termId = self.getValue()
        # Lowercasing the key to the term in the frequency list
        self._termId = self._termId.lower()
 
    def getTermType(self):
        return self.termType

       
    def isCompoundTerm (self):
        return self.termType != TermConstants['TYPE_SINGLE_TERM']

       
    def toString(self):
        return self.getValue()
 
       
    def valueEquals(self, term ):
        return self.toString() == term.toString()
    
    def valueEqualsIgnoreCase(self, term ):
        return self.toString().lower() == term.toString().lower()

class TagSet:
    def __init__(self):
        self.tags =[] #array
        self.TAG_SEPARATOR = ', '
        
    def addTag(self, term ):
        self.tags.append( term )
       
    def addAllTags(self, tagArray ):
        self.tags = self.tags.extend( tagArray )
       
    def getTags(self):
        return self.tags
       
    def toString(self, separator=None ):
        if separator != None:
            self.TAG_SEPARATOR = separator
     
        return self.tags.join( self.TAG_SEPARATOR )
       
    def sortByScore(self):
        self.tags.sort( self._scoreComparator )
       
    def _scoreComparator(self, a, b ):
        return int(b.getScore() - a.getScore())

    def toList(self):
        l = []
        for t in self.tags:
            l.append(t.getValue()+',')
        return l





class Tagger():
    def __init__(self):
        self.REMOVE_SHORT_NUMBERS_AS_SINGLE_TOKENS = True # Remove all numbers with 4 digits or less
        self.LOWERCASE = True # If true all terms are lowercased before returning
        self.EXTRACT_SPECIAL_TERMS = True # Extract abbreviations, acronyms and CamelCase words.
       
        self.TOKEN_LENGTH_CUTOFF = 2 # Only consider single tokens that are longer than n characters
        self.TERM_FREQUENCY_CUTOFF = 1 # Ignore terms that have fewer than n occurrences
        self.SCORE_CUTOFF = 0 # Ignoring terms that score less than n

        self.SINGLE_TERM_BOOST = 0.75
        self.WHITE_LIST_BOOST = 1.5 # This boost is applied to all words found in the white list
        self.CAPITALIZATION_BOOST = 1.75; # This boost is applied once to capitalised tokens, and again if all caps
        self.NGRAM_BASED_ON_CAPITALISATION_BOOST = 3.5 # This boost is applied to capitalised bi- and trigrams
        self.SPECIAL_TERM_BOOST = 2.5 # This boost is applied to capitalised bi- and trigrams
        self.BIGRAM_BOOST = 2.5 # This is applied to bigrams that do not contain stopwords and whose individual tokens are longer than 2 characters
        self.BIGRAM_ALREADY_DETECTED_BOOST = 0.25 # This boost is applied to all bigrams found to be wholly contained within a compound term detected based on capitalisation
        self.TERM_FROM_COMPOUND_DOWNWEIGHT = 0.25 # This is applied to individual tokens within an n-gram (every time an n-gram is discovered)
       
        self.COMPOUND_TAG_SEPARATOR = AUTOTAGS['DEFAULT_COMPOUND_TAG_SEPARATOR'] # Intra-tag (e.g. cool_gadget vs. cool gadget) separator to use
       
        # Remove all whitespace characters (certain white space characters are turned into boundaries)
        self.WHITESPACE_EXPRESSION = re.compile(r"(\')?([^a-zA-Z0-9_\.\!\?\:\;\n\r\f\t])")
        # Look for compound terms (bi- and trigrams) based on capitalization, accounting for corner cases like PayPal, McKinley etc.
        # TODO Need to estimate whether this is too greedy or not
        self.CAPITALIZED_NGRAM_EXPRESSION = re.compile(r"(([A-Z][a-z]*)?[A-Z][a-z]+ (of )?(Mc|Mac)?[A-Z][a-z]+([ \-][A-Z][a-z]+)?([ ][A-Z][a-z]+)?)")
        # Special Terms Expression to extract e.g. abbreviations and acronyms (with support for CamelCase words like JavaScript)
        self.SPECIAL_TERMS_EXPRESSION = re.compile(r"\b([A-Za-z]{1,2}\-[A-Za-z]+)|(([A-Z]\.){2,})|((([A-Z][A-Z0-9\-\:\_\+]+)|([A-Z]+[a-z]*?[A-Z][a-z]*?))( [A-Z][A-Za-z]+)?( [A-Z][A-Za-z]+)?( [0-9]*(\.[0-9]*)?)?)\b")
        # This expression looks for 'short numbers' with less than four digits (this will be included in stopword expression)
        self.SHORT_NUMBERS_EXPRESSION = '[0-9]{1,3}'
       
        # This is the whitelist cache
        self.whitelistCache = {}
        # Tag constants
        self.tagConstants = None

    def analyse_text(self, text, numberOfTagsToReturn ):
        # Starting
        startTime = datetime.datetime.now()


        # Data Structures
        frequencyListSingleTerms = FrequencyList()
        frequencyListCapitalisedCompoundTerms = FrequencyList()
        frequencyListSimpleBigramTerms = FrequencyList()
        frequencyListSpecialTerms = FrequencyList()
       
        # Instance Variables
        algorithmTime = 0

       	# Preprocessing text
       
        # Replacing all whitespace characters with a single space
        textWithWhitespaceRemoved =self.WHITESPACE_EXPRESSION.sub(' ',' ' + text + ' ') #check
       
        # Swapping certain punctuation for a boundary marker
        textWithBoundaryMarkers =re.sub( "([ ]*[\.\!\?\:\;\n\r\f\t][ ]*)+",' ' + AUTOTAGS['BOUNDARY'] + ' ',textWithWhitespaceRemoved) #check
        
        # Removing stopwords
        textWithWhitespaceAndStopwordsRemoved =self._getStopWordRegExpression().sub(' ', textWithBoundaryMarkers) #check

        # Splitting tokens into individual terms
        tokensToProcess = textWithWhitespaceAndStopwordsRemoved.split(' ')
        
   

        """
        
         1st Pass (building the frequency list)
        
        """
       
        # Identifying all single term candidates
        nums = range(len(tokensToProcess))
        for i in nums:
            token = tokensToProcess[i];
            if len(token) > self.TOKEN_LENGTH_CUTOFF:
                term = Term()
                term.setBoost(self.SINGLE_TERM_BOOST)
                term.setValue( token )
                term.ignoreTermFreqCutoff = False

                # Adding the candidate to the frequency list
                frequencyListSingleTerms.addTerm( term )
                
        
     
        # Identifying all special terms
        if self.EXTRACT_SPECIAL_TERMS:
            specialTerms = self.SPECIAL_TERMS_EXPRESSION.findall( text );
           
            if specialTerms != None :
                for special_term in specialTerms:
                    term = Term()
                    term.setTermType(TermConstants['TYPE_SPECIAL_TERM'])
                    term.setBoost(self.SPECIAL_TERM_BOOST)
                    term.setValue( special_term[3].strip())
                    term.ignoreTermFreqCutoff = True;
                    # Adding the candidate to the frequency list
                    frequencyListSpecialTerms.addTerm( term )
                    

        
        # Identifying compound terms based on capitalization
        capitalizedNGrams = self.CAPITALIZED_NGRAM_EXPRESSION.match( textWithBoundaryMarkers );
       
        if capitalizedNGrams != None:
                
                for capitalizedNGram in capitalizedNGrams:
                    compoundTermValue = capitalizedNGram

                    # The compound term should not start with a word from the blacklist, I try removing it and see what I'm left with.
                    compoundTermArray = compoundTermValue.split(' ')
                    if self.isInBlackList( compoundTermArray[0] ):
                        compoundTermValue = compoundTermValue.substr( compoundTermValue.indexOf(' ') + 1 )
                    
                   
                    term = Term()
                    term.setTermType(TermConstants['TYPE_CAPITALISED_COMPOUND_TERM'])
                    term.setBoost(self.NGRAM_BASED_ON_CAPITALISATION_BOOST)
                    term.setValue( compoundTermValue )
                    term.ignoreTermFreqCutoff = True

                    # Adding the candidate to the frequency list
                    frequencyListCapitalisedCompoundTerms.addTerm( term )

        # Identifying bi-grams in the text
        bigrams = textWithBoundaryMarkers.split(' ')
        nums = range(len(bigrams))
        
        for i in nums:
            position = i
            
            token1 = bigrams[position]
            if position +1 <= len(bigrams)-1:
                token2 = bigrams[position + 1]
            else:
                token2 = None
        
            if token1 != None and token2 != None and (len(token1) > 2 and len(token2) > 2 ) and self.isInBlackList(token1) == False and self.isInBlackList(token2) == False:
                bigram = token1 + ' ' + token2
                term = Term()
                term.setTermType(TermConstants['TYPE_SIMPLE_BIGRAM_TERM'])
                term.setBoost(self.BIGRAM_BOOST)
                term.setValue( bigram )
                term.ignoreTermFreqCutoff = False
               
                # Adding the candidate to the frequency list
                frequencyListSimpleBigramTerms.addTerm( term )
                




        
        """
        
         2nd Pass (evaluation and scoring of individual and compound terms)
        
        """
        
        temporaryTagSet = TagSet();
       
        # The order in which the frequency lists are analyzed is important!!!
        frequencyLists = [ frequencyListSpecialTerms, frequencyListCapitalisedCompoundTerms, frequencyListSimpleBigramTerms, frequencyListSingleTerms ];
        nums = range(len(frequencyLists))
        for listId in nums:
            listBeingProcessed = frequencyLists[listId];
           
            # Analyzing all terms within the list
            for termId in listBeingProcessed.getTerms():
                term = listBeingProcessed.getTermById( termId )
                ignoreTerm = False;

                if  (term.freq > self.TERM_FREQUENCY_CUTOFF) or (self.isInWhiteList(term.getValue()) or term.ignoreTermFreqCutoff == True):
                   
                            """
                             Filtering...removing obvious duplicate terms between across lists and deciding between capitalised
                             compound terms and bigrams (for which there might exist corresponding entries in both lists)
                            """
                            if term.termType == TermConstants['TYPE_SPECIAL_TERM']:
                                ignoreSpecialTerm = False;
                                # Process all frequency lits but the special term one...
                                # check
                                for specialTermLookupListId in nums:
                                        specialTermLookupList = frequencyLists[specialTermLookupListId];
                                        if specialTermLookupList == listBeingProcessed:
                                            # This is the special term list, move on...
                                            continue
                                        else:
                                            # Checking if this special term exists in the list being processed
                                            termToLookup = term.getTermId();
                                            # I'm maybe being to greedy here - if the special term doesn't exist in it's natural form in the single term list I try stemming it...
                                            if specialTermLookupList == frequencyListSingleTerms and specialTermLookupList.getTermById( termToLookup ) == None:
                                                    termToLookup = _stemToken(termToLookup)
                                            
                                            if specialTermLookupList.getTermById( termToLookup ) != None:
                                                specialTermInList = specialTermLookupList.getTermById( termToLookup );
                                                # If a more frequent or higher scoring variant of the special term is found in one of the other lists then ignore this one
                                                if specialTermInList.getScore() > term.getScore():
                                                        ignoreTerm = True
                                                        continue
                                                else:
                                                        # The special term is more frequent or higher scoring...so delete from the other list
                                                        specialTermLookupList.deleteTermById( termToLookup );

                            elif term.termType == TermConstants['TYPE_CAPITALISED_COMPOUND_TERM']:
                                """
                                 Checking if term is TYPE_CAPITALISED_COMPOUND_TERM!
                                 These capitalised compounds require special handling. If they exist in the bigram frequency list, they are clearly
                                 bigrams and therefore we should consider which frequency number to use, since there clearly might be more instances if
                                 we ignore case.
                                """ 
                                bigram = frequencyListSimpleBigramTerms.getTermById( term.getTermId() )
                               
                                if bigram != None:
                                        # The capitalised compound term exists as a bigram
                                        if bigram.freq > term.freq:
                                            # There are more bigram variants than compound ones. I will therefore ignore the compound one since
                                            # it may e.g. have been capitalised in a title.
                                            # Adding a boost to the upcoming bigram variant since it's clearly more than just a normal bigram
                                            bigram.addBoost( self.CAPITALIZATION_BOOST )
                                            ignoreTerm = True
                                        else:
                                            # There is an equal or less number of bigrams, therefore I remove the bigram and go with the capitalised variant
                                            frequencyListSimpleBigramTerms.deleteTermById( term.getTermId() )

                               
                                # Now checking if it exists as a simple term (that might happen if I remove blacklisted word at the front)
                                simpleTerm = frequencyListSingleTerms.getTermById( term.getTermId() )
                                if not ignoreTerm and simpleTerm != None:
                                    if simpleTerm.getScore() > term.getScore():
                                            simpleTerm.addBoost( self.CAPITALIZATION_BOOST )
                                            ignoreTerm = True
                                    else:
                                            frequencyListSingleTerms.deleteTermById( term.getTermId() )

                           
                            """
                             Calculating initial boosts
                            """
                   
                            # Term is in the whitelist
                            if self.isInWhiteList( term.getValue() ): 
                                term.addBoost( self.WHITE_LIST_BOOST )
                            if not term.isCompoundTerm():
                                # Term is capitalized
                                if term.getValue()[0].upper() == term.getValue()[0]:
                                        term.addBoost( self.CAPITALIZATION_BOOST )
                            
                                # Term is all in caps (double boost)
                                if term.getValue().upper() == term.getValue():
                                        term.addBoost( self.CAPITALIZATION_BOOST )
                                
                            
                   
                            # Lowercasing the word if specified by the LOWERCASE parameter
                            if self.LOWERCASE:
                                term.setValue( term.getValue().lower() )
                            

                            # Adding the term to final stage evaluation if it meets the SCORE_CUTOFF criteria
                            if not ignoreTerm and term.getScore() > self.SCORE_CUTOFF:
                                    temporaryTagSet.addTag( term )
   
       
       
       
        """
        
         3rd Pass (Order based on score and further honing based on that order)
        
        """
        
        # Sorting the TagSet array by score
        temporaryTagSet.sortByScore()
       
        # Final TagSet to be returned
        tagSetToBeReturned = TagSet();
       
        # This array will hold bigrams of the detected compound terms for quick lookup when general bigrams are detected
        temporaryBigramArrayOfCapitalizedNGrams = []
        temporaryArrayOfSplitBigrams = []
       
        for  t in temporaryTagSet.tags:
                term = t #temporaryTagSet.tags[t];
                
               
                if term.termType == TermConstants['TYPE_CAPITALISED_COMPOUND_TERM']:
                        # Checking if term is TYPE_CAPITALISED_COMPOUND_TERM
                        # Adding a bigram of it to a temporary array
                        temporaryBigramArrayOfCapitalizedNGrams = temporaryBigramArrayOfCapitalizedNGrams.concat( self._toBigramArray( term.getValue().lower() ) )
                        # Adding compound term components to a separate array to downweight single terms found within a
                        # higher scoring compound term
                        capitalisedCompoundTermComponents = term.getValue().lower().split(' ')
                        it = range(len(capitalisedCompoundTermComponents))
                        for t2 in it:
                            tokenToAdd = capitalisedCompoundTermComponents[t2] 
                            if AUTOTAGS['APPLY_STEMMING']:
                                tokenToAdd = _stemToken( tokenToAdd ) 
                            temporaryArrayOfSplitBigrams.push( tokenToAdd )
                        
                elif  term.termType == TermConstants['TYPE_SIMPLE_BIGRAM_TERM']:
                        # If this bigram exists in the array of 'bigrams made from capitalised compound terms' it means that
                        # the capitalised compound term is higher scoring (since it went before) and therefore should the simple
                        # bigram which is      contained wholly within the capitalised compound term be downweighted.
                        if _arrayContains( temporaryBigramArrayOfCapitalizedNGrams, term.getValue().lower() ):
                            term.addBoost( self.BIGRAM_ALREADY_DETECTED_BOOST )
                        
                       
                        # Adding bigram components to a separate array to downweight single terms found within a
                        # higher scoring bigram
                        bigramComponents = term.getValue().lower().split(' ')
                        it = range(len(bigramComponents))
                        for t2 in it:
                            bigramTokenToAdd = bigramComponents[t2] 
                            if AUTOTAGS['APPLY_STEMMING']:
                                bigramTokenToAdd = _stemToken( bigramTokenToAdd )  
                            temporaryArrayOfSplitBigrams.append( bigramTokenToAdd )
                        
                       
                elif term.termType == TermConstants['TYPE_SINGLE_TERM']:
                        # Checking if this simple term is found within a higher scoring bigram
                        # If it is found in the temporary array of split bigrams it means that it has a lower score
                        # since the bigram was processed before it.
                        if AUTOTAGS['APPLY_STEMMING']:
                            termValue = _stemToken(term.getValue())
                        if _arrayContains( temporaryArrayOfSplitBigrams, termValue ):
                            term.addBoost( self.TERM_FROM_COMPOUND_DOWNWEIGHT )
                        
                
                # TODO check..
                #if self.COMPOUND_TAG_SEPARATOR != AUTOTAGS['DEFAULT_COMPOUND_TAG_SEPARATOR']:
                #        term.setValue( term.getValue().replace(/ /g, self.COMPOUND_TAG_SEPARATOR ) )
                #}
               
                tagSetToBeReturned.addTag( term )
        
       
        # Cleaning up...
        # TODO check if needed 
        # temporaryBigramArrayOfCapitalizedNGrams.length = 0;
        # temporaryArrayOfSplitBigrams.length = 0;
        # temporaryTagSet.length = 0;
       
        # Sorting the TagSet array by score
        tagSetToBeReturned.sortByScore();
       
        # Slicing out top tags to return
        # TODO fix slicing..
        #tagSetToBeReturned.tags = tagSetToBeReturned.tags.slice( 0, numberOfTagsToReturn )    
        #tagSetToBeReturned.addAllTags( self.getTagConstants() )
       
        # Done
        self._setAlgorithmTime(datetime.datetime.now() -startTime )
       
        return tagSetToBeReturned        
        
    

    def getTagConstants(self):
        if self.tagConstants != None:
            return self.tagConstants
        else:
            self.tagConstants = []
               
            if TAG_CONSTANTS != None:
                for tag_constant in TAG_CONSTANTS:
                    constant = Term()
                    constant.setTermType(TermConstants['TYPE_TAG_CONSTANT'])
                    constant.setValue(tag_constant)
                    self.tagConstants.append( constant )
                        
        
               
            return self.tagConstants
        

       
    def _toBigramArray(self, compoundTerm ):
        bigramArray = []
       
        splitTerm = compoundTerm.split( ' ' )
        nums = range(len(splitTerm))
        for num in nums:
                position = num
                token1 = splitTerm[position]
                token2 = splitTerm[position + 1]
                if token1 != None and token2 != None:
                    bigramArray.push( token1 + ' ' + token2 )
        return bigramArray
       
    def isInWhiteList(self, term ):
                # Whitelist lookup with caching
                # In case the same words are prevalent in the text I can avoid looking them up again
                if term in self.whitelistCache:
                        return self.whitelistCache[term]
                else:
                    inWhiteList = False
                    if WHITELIST != None:
                        inWhiteList = _arrayContains( WHITELIST, term.lower() )
                        self.whitelistCache[term] = inWhiteList
                                
                    return inWhiteList



    def isInBlackList(self, term ):
        if term != AUTOTAGS['BOUNDARY'] and ( STOPWORDS != None and not term.lower() in STOPWORDS ):
            return False
        else:
            return True

       
    def _getStopWordRegExpression(self):
        blacklistExpression = AUTOTAGS['BOUNDARY']
       
        # Adding all blacklist terms
        if STOPWORDS != None:
            for term in STOPWORDS:
                blacklistExpression += '|' + term


        if self.REMOVE_SHORT_NUMBERS_AS_SINGLE_TOKENS:
            return re.compile( '\\s((' + self.SHORT_NUMBERS_EXPRESSION + '|' + blacklistExpression + ')\\s)+')
        else:
            return re.compile( '\\s((' + blacklistExpression + ')\\s)+' )

               
    def getAlgorithmTime(self):
        return self.algorithmTime
       
    def _setAlgorithmTime(self, timeInMilliseconds ):
        self.algorithmTime = timeInMilliseconds


"""

       Frequency List Business Object

"""
class FrequencyList:
    def __init__(self):
        self._terms = {}

    def addTerm(self, term ):
        # Is the term in the frequency list? If so then retrieve it and increment frequency
        if self.getTermById( term.getTermId() ) != None:
            # Getting only frequency from the existing term, updating everything else
            term.freq = (self.getTermById( term.getTermId() ).freq + 1)
               
        # Updating frequency list with the term being processed
        self._terms[term.getTermId()] = term
        
       
    def getTermById(self, termId ):
        if termId in self._terms:
            return self._terms[ termId ]
        else:
            return None
       
    def getTerms(self):
        return self._terms
       
    def deleteTermById (self, termId ):
        try:
            del self._terms[termId]
        except KeyError:
            pass





"""
       Get the root of a given word
"""
def _stemToken( token ):
    token = token.lower()
    # Find the root of words and cache since stemming is fairly expensive in this context
    if  token in VARIATION_CACHE:
        return VARIATION_CACHE[token]
    else:
        # Token not in the cache, stemming and adding to the cache
       
        stemmer_impl = _get_stemmer_impl()
        stemmed_variant = stemmer_impl(token)
        VARIATION_CACHE[token] = stemmed_variant
        return stemmed_variant


STEMMER = PorterStemmer()
def _get_stemmer_impl():
    return STEMMER.stem_word


def _arrayContains(array,o):
    for obj in array:
        if o == obj:
            return True
    return False

# This is a cache of all root words (stemmed variants) for quick lookup (stemming is fairly expensive in this context)
VARIATION_CACHE = {}
