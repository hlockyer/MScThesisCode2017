The code given within this repository is based upon a Python implementation of the Constraints on New Theories Using Rivet (CONTUR) methodology, written by Professor Jonathan Butterworth and David Yallup at University College London.

The files above are elements of the core statistics code, responsible for implementing limit setting on new physical models via the statistical method outlined by Butterworth et al in the following reference paper:
https://arxiv.org/abs/1606.05296 (Latest revision: March 6th 2017)
As such, elements of the code found within the files above have remained unchanged from their original versions. The intention of this thesis was to build upon the existing code and software framework in an attempt to facilitate the inclusion of simulated Standard Model data that is not taken to be equal to background event counts.

Modifications have been made to the content of TesterFunctions.py with CONTUR - these are reflected in the new version, TesterFunctionsUpdated.py. The original file has also been included for reference. 
No modifications have been made to the confidence level script, CLTestSingle.py. This file has been included as a reference for the fact that the form of the second derivatives found within TesterFunctionsUpdated.py takes 'sgError' as the term for the ratio of the Monte Carlo luminosity values to the data luminosity values - the origin of this methodology can be found within CLTestSingle.py.
