
from utils import *


row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
unitlist = row_units + column_units + square_units
diag_units = [[rs+cs for (rs,cs) in zip(rows, cols)], [rs+cs for (rs,cs) in zip(rows, cols[::-1])]]
# TODO: Update the unit list to add the new diagonal units
unitlist = row_units + column_units + square_units + diag_units


# Must be called after all units (including diagonals) are added to the unitlist
units = extract_units(unitlist, boxes)
peers = extract_peers(units, boxes)


def naked_twins(values):
    """Eliminate values using the naked twins strategy.

    The naked twins strategy says that if you have two or more unallocated boxes
    in a unit and there are only two digits that can go in those two boxes, then
    those two digits can be eliminated from the possible assignments of all other
    boxes in the same unit.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with the naked twins eliminated from peers

    Notes
    -----
    Your solution can either process all pairs of naked twins from the input once,
    or it can continue processing pairs of naked twins until there are no such
    pairs remaining -- the project assistant test suite will accept either
    convention. However, it will not accept code that does not process all pairs
    of naked twins from the original input. (For example, if you start processing
    pairs of twins and eliminate another pair of twins before the second pair
    is processed then your code will fail the PA test suite.)

    The first convention is preferred for consistency with the other strategies,
    and because it is simpler (since the reduce_puzzle function already calls this
    strategy repeatedly).

    See Also
    --------
    Pseudocode for this algorithm on github:
    https://github.com/udacity/artificial-intelligence/blob/master/Projects/1_Sudoku/pseudocode.md
    """
    twins_dict = {} # dictionary that contains all the naked twins in the puzzle
                    # example is ('B7', 'A7'): {'2', '7'}
    for unit in unitlist: # move through the unit lists 
        pairs = {box:values[box] for box in unit if len(values[box]) == 2} # contains the cells which are candidates for twin pairs
        pairs = {p: set(values[p]) for p in pairs} # convert the values to a set since the order doesn't matter 
        naked_twins = {} # reduce the set of candidate twin pairs to the set of all naked twins 
        for key,value in pairs.items(): 
            match = [k for (k, v) in pairs.items() if v == value and k != key] # checking to see if a twin exists! 
            if match: # a match exists iff there exists the same value with a different key! 
                naked_twins[match[0]] = value
        # We now merge the keys of the dictionary with common values into a tuple
        # For example, if naked_twins = {'E2': {'5', '7'}, 'E3': {'4', '8'}, 'E5': {'5', '7'}, 'E8': {'4', '8'}}
        # Then the output (twins_merge) should look like =  {('E2', 'E5'): {'5', '7'}, ('E3', 'E8'): {'4', '8'}}
        naked_twins_temp = defaultdict(list) # Store the naked twins in a list 
        for a, b in naked_twins.items():
            naked_twins_temp[tuple(b)].append(a)
        twins_merge = {tuple(b): set(a) for a, b in naked_twins_temp.items()}
        twins_dict.update(twins_merge) # Update the full dictionary of all naked twins in the puzzle 

    # Eliminate the naked twins as possibilities for their peers
    for twins, v in twins_dict.items():
        v = list(v) # convert from set to list 
        # We now compute the candidates that may contain values that we can eliminate
        eliminate_candidates = [set(peers[twins[0]]).intersection(peers[twins[1]])][0]
        for e in eliminate_candidates:
            if v[0] in values[e]: # check if the younger twin can be crossed out 
                values = assign_value(values, e, values[e].replace(v[0], ''))
            if v[1] in values[e]: # check if the older twin can be crossed out 
                values = assign_value(values, e, values[e].replace(v[1], ''))
                
    return values


def eliminate(values):
    """Apply the eliminate strategy to a Sudoku puzzle

    The eliminate strategy says that if a box has a value assigned, then none
    of the peers of that box can have the same value.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with the assigned values eliminated from peers
    """
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
            values = assign_value(values, peer, values[peer].replace(digit,''))
    return values


def only_choice(values):
    """Apply the only choice strategy to a Sudoku puzzle

    The only choice strategy says that if only one box in a unit allows a certain
    digit, then that box must be assigned that digit.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with all single-valued boxes assigned

    Notes
    -----
    You should be able to complete this function by copying your code from the classroom
    """
    for u in unitlist:
        firstSeen = {}
        seen = []
        seenAlready = []
        for cell in u:
            if len(values[cell]) > 1:
                for num in values[cell]:
                    if num not in seen: # first time seeing value
                        seen.append(num)
                        firstSeen[num] = cell
                    else: # already seen
                        if num not in seenAlready: # only care if the val was seen more than onnce
                            seenAlready.append(num)
        for seen_value in seen:
            if seen_value not in seenAlready and seen_value not in [i for i in list({k: values[k] for k in u}.values()) if len(i)==1]:
                #print(firstSeen[seen_value], seen_value)
                values = assign_value(values, firstSeen[seen_value], seen_value)
    return values


def reduce_puzzle(values):
    """Reduce a Sudoku puzzle by repeatedly applying all constraint strategies

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict or False
        The values dictionary after continued application of the constraint strategies
        no longer produces any changes, or False if the puzzle is unsolvable 
    """
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        # Use the Eliminate Strategy
        values = eliminate(values)
        # Use the Only Choice Strategy
        values = only_choice(values)
        # Use the Naked Twins Strategy 
        values = naked_twins(values)
        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values


def search(values):
    """Apply depth first search to solve Sudoku puzzles in order to solve puzzles
    that cannot be solved by repeated reduction alone.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict or False
        The values dictionary with all boxes assigned or False

    Notes
    -----
    You should be able to complete this function by copying your code from the classroom
    and extending it to call the naked twins strategy.
    """
    values = reduce_puzzle(values)
    if values is False:
        #print("Values is False")
        return False
    if all(i==1 for i in [len(values[k]) for k in values]):
        #print('Done!!!')
        return values
    # Choose one of the unfilled squares with the fewest possibilities
    candidate_squares = {key : len(value) for (key, value) in values.items() if len(value)>1}
    square_min = min(candidate_squares, key=candidate_squares.get)
    childQueue = [] # Create a queue of children puzzle 
    for num in values[square_min]:
        childDict = {k:v for k, v in values.items()}
        childDict = assign_value(childDict, square_min, num)
        childQueue.append([childDict, [square_min, num]])
        
    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
    for child, change in childQueue:
        #print(change[0], change[1], 'from', values[square_min])
        solution = search(child)
        if solution != False: # if a child is not False, it might be able to be solved 
            # This child is possibly solvable 
            return solution
    # if we make it thru the for loop then every child is False, which means we can prune that section of the tree
    return False


def solve(grid):
    """Find the solution to a Sudoku puzzle using search and constraint propagation

    Parameters
    ----------
    grid(string)
        a string representing a sudoku grid.
        
        Ex. '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'

    Returns
    -------
    dict or False
        The dictionary representation of the final sudoku grid or False if no solution exists.
    """
    values = grid2values(grid)
    values = search(values)
    return values


if __name__ == "__main__":
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(grid2values(diag_sudoku_grid))
    result = solve(diag_sudoku_grid)
    display(result)

    try:
        import PySudoku
        PySudoku.play(grid2values(diag_sudoku_grid), result, history)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
