"""
TAO represents the change in volatility over time. 

"Smaller values of TAO prevent the volatility measures from changing by large amounts, 
which in turn prevent emormous changes in ratings based on
very improbable results"
    - Example of the Glick-2 System by Mark E. Glickman (2022)
"""

TAO = 0.5
assert 0.2 <= TAO and TAO <= 1.2, "Tao is not a reasonable number"

INITIAL_RATING = 1500
INITIAL_DEVIATION = 350
INITIAL_VOLATILITY = 0.06
