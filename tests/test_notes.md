# main issues
- Semantic error count does not corespond to the number of visible word-level replacement errors in the text. 
- Document score does not reflect the severity of the error, in some cases a single word error can lead to a very low document score, but in other cases multiple word errors can lead to a high document score.

# Punctuation and spacing should not register as semantic error 
```bash
Macleod , is insisting on a policy of change .	
Macleod, is insisting on a policy of change.
```
doc_score 0.446579	
semantic_error_count 2


# Good example of semantic error, incorrect word, but is a viable real word, semantic error count is 3. Doc score is low and does not reflect the severity of the error.
```
chief aide , Mr. Julius Greenfield , telephoned	
chief vide, mr. julius greenfield, telephoned
```
doc_score 0.406974	
semantic_error_count 3

# Good example of semantic error, first error "Mr" leads to a name "Komarov" rather than "tomorrow"
```
meeting of Labour 0M Ps tommorow . Mr. Michael	
meeting of Labour 017 Mr Komarov. Mr. Michael
```
doc_score 0.476989		
semantic_error_count 3

# Notable high doc score, domain-specific term, language shift (Labour/Calour)?
```
Peeresses have been created . Most Labour	
Peresses have been created. Most Calour
```
doc_score 0.665275	    
semantic_error_count 3

# OCR-style single charachter error 
```
Mr. Iain Macleod , is insisting on a policy of...	
Mr. lain Hacleod, is insisting on a policy of ...
```
doc_score 0.435457    
semantic_error_count 3


# After updates 
Good 
be that as Labour M Ps opposed the	
be that as Labour MPs opposed me

Just char error, not semantic
National Independence Party ( 280,000 members )	
National Independence Party (280'000 member)

Good, interesting that Peress is a real family name
Peeresses have been created . Most Labour	
Peresses have been created. Most Calour

Good example of semantic error (gather/garner)
Though they may gather some Left-wing	
Though they may garner some left-wing