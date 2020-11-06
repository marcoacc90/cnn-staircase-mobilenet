import sys
sys.path.insert(1, './../' )
import Global as G

def loadNames( file ) :
    f = open( file, 'r' )
    e = '/' + str( G.EPOCHS )
    for line in f :
        row = line.split()
        if( len(row) != 2 and len(row) != 17 ) :
            continue;
        for i in range( len(row ) ) :
            if row[ i ] ==  'Epoch' :
                epoch = row[ i + 1 ].replace( e, '' )
                print( int( epoch ), end= ' ' )
            elif row[ i ] == 'loss:' :
                print( float(row[ i + 1 ] ), end= ' ' )
            elif row[ i ] == 'val_loss:' :
                print( float(row[ i + 1 ] ), end= ' ' )
            elif row[ i ] ==  'accuracy:' :
                print( float(row[ i + 1 ] ), end= ' ' )
            elif row[ i ] == 'val_accuracy:' :
                print( float(row[ i + 1 ] ) )
    f.close()
    return 0

if len( sys.argv ) != 2:
    print ("python3 formatTrainEvolution <input_file> > <output_file> ")
    print ("python3 formatTrainEvolution ./../models/mobilenet_train_evo_raw.txt > ./../models/mobilenet_train_evo_raw.txt")
    exit( 0 )


a = loadNames( sys.argv[ 1 ] )
