import warnings
warnings.filterwarnings("ignore")
import sys
import os
target = ""

if __name__ == "__main__":
    target = raw_input("Input Sentence:\n")
    observation = target.split()
    '''
    transition = { 'START' : {'NNP' : 0.2767, 'MD' : 0.0006, 'VB' : 0.0031, 'JJ' : 0.0453, 'NN' : 0.0449, 'RB' : 0.0510, 'DT' : 0.2026},
                          'NNP' : {'NNP' : 0.3777, 'MD' : 0.0110, 'VB' : 0.0009, 'JJ' : 0.0084, 'NN' : 0.0584, 'RB' : 0.0090, 'DT' : 0.0025},
                          'MD' : {'NNP' : 0.0008, 'MD' : 0.0002, 'VB' : 0.7968, 'JJ' : 0.0005, 'NN' : 0.0008, 'RB' : 0.1698, 'DT' : 0.0041},
                          'VB' : {'NNP' : 0.0322, 'MD' : 0.0005, 'VB' : 0.0050, 'JJ' : 0.0837, 'NN' : 0.0615, 'RB' : 0.0514, 'DT' : 0.2231},
                          'JJ' : {'NNP' : 0.0366, 'MD' : 0.0004, 'VB' : 0.0001, 'JJ' : 0.0733, 'NN' : 0.4509, 'RB' : 0.0036, 'DT' : 0.0036},
                          'NN' : {'NNP' : 0.0096, 'MD' : 0.0176, 'VB' : 0.0014, 'JJ' : 0.0086, 'NN' : 0.1216, 'RB' : 0.0177, 'DT' : 0.0068},
                          'RB' : {'NNP' : 0.0068, 'MD' : 0.0102, 'VB' : 0.1011, 'JJ' : 0.1012, 'NN' : 0.0120, 'RB' : 0.0728, 'DT' : 0.0479},
                          'DT' : {'NNP' : 0.1147, 'MD' : 0.0021, 'VB' : 0.0002, 'JJ' : 0.2157, 'NN' : 0.4744, 'RB' : 0.0102, 'DT' : 0.0017}}
    '''
    a = [[0.2767, 0.0006, 0.0031, 0.0453, 0.0449, 0.0510, 0.2026],
         [0.3777, 0.0110, 0.0009, 0.0084, 0.0584, 0.0090, 0.0025],
         [0.0008, 0.0002, 0.7968, 0.0005, 0.0008, 0.1698, 0.0041],
         [0.0322, 0.0005, 0.0050, 0.0837, 0.0615, 0.0514, 0.2231],
         [0.0366, 0.0004, 0.0001, 0.0733, 0.4509, 0.0036, 0.0036],
         [0.0096, 0.0176, 0.0014, 0.0086, 0.1216, 0.0177, 0.0068],
         [0.0068, 0.0102, 0.1011, 0.1012, 0.0120, 0.0728, 0.0479],
         [0.1147, 0.0021, 0.0002, 0.2157, 0.4744, 0.0102, 0.0017]]
    '''
    observation_likelihood = { 'NNP' : {'Janet' : 0.000032, 'will' : 0, 'back' : 0, 'the' : 0.000048, 'bill' : 0},
                           'MD' : {'Janet' : 0, 'will' : 0.308431, 'back' : 0, 'the' : 0, 'bill' : 0},
                           'VB' : {'Janet' : 0, 'will' : 0.000028, 'back' : 0.000672, 'the' : 0, 'bill' : 0.000028},
                           'JJ' : {'Janet' : 0, 'will' : 0, 'back' : 0.000340, 'the' : 0, 'bill' : 0},
                           'NN' : {'Janet' : 0, 'will' : 0.000200, 'back' : 0.000223, 'the' : 0, 'bill' : 0.002337},
                           'RB' : {'Janet' : 0, 'will' : 0, 'back' : 0.010446, 'the' : 0, 'bill' : 0},
                           'DT' : {'Janet' : 0, 'will' : 0, 'back' : 0, 'the' : 0.506099, 'bill' : 0}}
    '''

    b = [ [0.000032, 0, 0, 0.000048, 0],
          [0, 0.308431, 0, 0, 0],
          [0, 0.000028, 0.000672, 0, 0.000028],
          [0, 0, 0.000340, 0, 0],
          [0, 0.000200, 0.000223, 0, 0.002337],
          [0, 0, 0.010446, 0, 0],
          [0, 0, 0, 0.506099, 0]]

    states = ['NNP','MD','VB','JJ','NN','RB','DT']
    words= ['Janet', 'will', 'back', 'the', 'bill']
    viterbi = list()
    back = list()
    T = len(observation)
    N = len(states)

    for i in range(N+2):
        viterbi.append(list())
        back.append(list())
        for j in range(T):
            viterbi[i].append(-1)
            back[i].append(-1)

    for s in range(N):
        viterbi[s][0] = a[0][s]*b[s][words.index(observation[0])] 
        back[s][0] = 0
    for t in range(1, T):
        for s in range(N):
            max = -1.0
            back_state = -1.0
            temp = 0.0
            for k in range(N):
                temp = float(viterbi[k][t-1]) *float(a[k+1][s])*float(b[s][words.index(observation[t])])
                if temp > max:
                    max = temp
                    back_state = k
            viterbi[s][t] = max
            back[s][t] = back_state

    max = -1.0
    st = 0
    for s in range(N):
        if max< viterbi[s][T-1]:
            max = viterbi[s][T-1]
            st = s
    viterbi[N][T-1] = max
    back[N][T-1] = st
    start = back[N][T-1]
    tags = list()
    tags.append(start)
    for ind in range(1,T):
        val = back[start][T-ind]
        tags.append(val)
        start = val
    tags.reverse()
    i = 0
    ans = list()
    for each in observation:
        ans.append(each+"/"+states[tags[i]])
        i = i + 1
    print(ans)
    print("Probability: ", max)





