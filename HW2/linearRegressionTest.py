import unittest
import linearRegression

class linearRegressionTest(unittest.TestCase):

    def test_error(self):
        with self.assertRaises(TypeError):
            linearRegression('test.csv') #checks input type in document correct or not  

if __name__ == '__main__': #Add this if you want to run the test with this script.
  unittest.main()
