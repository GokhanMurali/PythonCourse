import unittest
import linearRegression

class linearRegressionTest(unittest.TestCase):

    def test_error(self): #checks input type in document correct or not
        with self.assertRaises(TypeError):
            linearRegression('test.csv') #there are strings in this file. My code has to return TypeError

    def test_beta(self): #checks beta calculated true or not. I did regression analysis in excel with this file. I get these values from excel.
        beta, standard_deviations, confidence_intervals = linearRegression.linearRegression('SLR_Salary_Years in class.csv')
        self.assertEqual(28394.158061953924,beta[0][0]) #checks beta zero
        self.assertEqual(1107.218377158146,beta[1][0]) #checks beta one

    def test_standard_deviations(self): #checks standard_deviations calculated true or not. I did regression analysis in excel with this file. I get these values from excel.
        beta, standard_deviations, confidence_intervals = linearRegression.linearRegression('SLR_Salary_Years in class.csv')
        self.assertEqual(1793.9512498846082,standard_deviations[0]) #checks standard_deviation for beta zero
        self.assertEqual(140.447561940155,standard_deviations[1]) #checks standard_deviation for beta one

    def test_confidence_intervals(self): #checks confidence_intervals calculated true or not. I did regression analysis in excel with this file. I get these values from excel.
        beta, standard_deviations, confidence_intervals = linearRegression.linearRegression('SLR_Salary_Years in class.csv')
        self.assertEqual(24778.686890931698,confidence_intervals[0][0]) #checks lower limit of confidence_interval for beta zero
        self.assertEqual(32009.62923297615,confidence_intervals[0][1]) #checks upper limit of confidence_interval for beta zero
        self.assertEqual(824.1649155506186,confidence_intervals[1][0]) #checks lower limit of confidence_interval for beta zero
        self.assertEqual(1390.2718387656735,confidence_intervals[1][1]) #checks lower limit of confidence_interval for beta one

if __name__ == '__main__': #Add this if you want to run the test with this script.
  unittest.main()
