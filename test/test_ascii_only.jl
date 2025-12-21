using Test

@testset "ASCII-only source tree" begin
    function jl_files_under(dir::AbstractString)
        files = String[]
        for (root, _, fs) in walkdir(dir)
            for f in fs
                endswith(f, ".jl") || continue
                push!(files, joinpath(root, f))
            end
        end
        sort!(files)
        return files
    end

    function first_nonascii_byte(path::AbstractString)
        data = read(path)
        for (i, b) in enumerate(data)
            if b > 0x7f
                return (i, b)
            end
        end
        return nothing
    end

    src_dir  = normpath(joinpath(@__DIR__, "..", "src"))
    test_dir = normpath(@__DIR__)

    for f in vcat(jl_files_under(src_dir), jl_files_under(test_dir))
        bad = first_nonascii_byte(f)
        if bad !== nothing
            pos, byte = bad
            @info "Non-ASCII byte detected" file=f pos=pos byte=byte
        end
        @test bad === nothing
    end
end
