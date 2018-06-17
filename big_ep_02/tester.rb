num_mat = 500
size_mat = [3, 3]
range = [-100, 100]

out_name = 'out'
out_name = ARGV[0] if ARGV[0]

matrices = Array.new(3) { Array.new(3) { Array.new(num_mat) { rand range[0]..range[1] } } }

def print_matrix(matrices, matrix, f)
    matrices.each do |row|
        row.each do |col|
            f.print col[matrix] if matrix >= 0
            f.print col.min if matrix < 0
            f.print " "
        end
        f.print "\n"
    end
end

open("#{out_name}.mat", 'w') { |f|
    f.puts num_mat

    (0..(num_mat - 1)).each do |matrix|
        f.puts "***"
        print_matrix(matrices, matrix, f)
    end
}

open("#{out_name}.ans", 'w') { |f|
    print_matrix(matrices, -1, f)
}