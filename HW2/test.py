{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 284,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 285,
   "metadata": {},
   "outputs": [],
   "source": [
    "class crossEntropyLogit:\n",
    "    p = []\n",
    "    def doForward(self, z, y):\n",
    "        z_max = np.max(z, axis=0)\n",
    "        print(np.max(z, axis=0))\n",
    "        z = z - z_max\n",
    "        exp_z = np.exp(z)\n",
    "        sum_z = np.sum(exp_z, axis=0)\n",
    "        self.p = exp_z / sum_z\n",
    "        log_p = np.log(self.p)\n",
    "        return (np.sum(np.multiply(y, log_p)) * -1) / y.shape[1]\n",
    "    def doBackward(self, y):\n",
    "        return (self.p - y) / y.shape[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 286,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[4.17022005e-01, 7.20324493e-01],\n",
       "       [1.14374817e-04, 3.02332573e-01],\n",
       "       [1.46755891e-01, 9.23385948e-02]])"
      ]
     },
     "execution_count": 286,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "CE = crossEntropyLogit()\n",
    "np.random.seed(1)\n",
    "z = np.random.rand(3, 2)\n",
    "z"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 287,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 0.],\n",
       "       [0., 1.],\n",
       "       [0., 0.]])"
      ]
     },
     "execution_count": 287,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = np.eye(3)[:, :2]\n",
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 288,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.417022   0.72032449]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "1.0437608174087478"
      ]
     },
     "execution_count": 288,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "J1 = CE.doForward(z, y)\n",
    "J1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 289,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1000.417022   1000.72032449]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "1.0437608174087274"
      ]
     },
     "execution_count": 289,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "J2 = CE.doForward(z + 1000, y)\n",
    "J2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 290,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.417022   0.72032449]\n",
      "[1000.417022   1000.72032449]\n",
      "1.0437608174087478 1.0437608174087274\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "CE=crossEntropyLogit()\n",
    "np.random.seed(1)\n",
    "z=np.random.rand(3,2)\n",
    "y=np.eye(3)[:,:2]\n",
    "J1=CE.doForward(z,y) # should be 1.04376...\n",
    "J2=CE.doForward(z+1000,y) # should be 1.04376...\n",
    "print(J1, J2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 291,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-0.29358105  0.22809874]\n",
      " [ 0.13604698 -0.34982719]\n",
      " [ 0.15753407  0.12172845]]\n"
     ]
    }
   ],
   "source": [
    "dz=CE.doBackward(y)\n",
    "print(dz)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
