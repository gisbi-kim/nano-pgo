def evaluate_jacobian(pose_i, pose_j, pose_ij_meas)
    import symforce 
    symforce.set_log_level("ERROR")
    print(f"symforce uses {symforce.get_symbolic_api()} as backend")
    import symforce.symbolic as sf
    from symforce.ops import StorageOps, GroupOps, LieGroupOps

    epsilon = 0.000001
    
    sf_ri = sf.V3.symbolic("ri") # i.e., angle-axis parametrization
    sf_Ri = LieGroupOps.from_tangent(sf.Rot3, sf_ri)

    sf_rj = sf.V3.symbolic("rj") # i.e., angle-axis parametrization
    sf_Rj = LieGroupOps.from_tangent(sf.Rot3, sf_rj) 

    sf_rij = sf.V3.symbolic("rij")
    sf_Rij = LieGroupOps.from_tangent(sf.Rot3, sf_rij)

    sf_R_err = sf_Rij.inverse() * sf_Ri.inverse() * sf_Rj
    sf_r_err = sf.Matrix(sf_R_err.to_tangent())

    sf_J_rerr_ri = sf_r_err.jacobian(sf_ri)
    sf_J_rerr_rj = sf_r_err.jacobian(sf_rj)

    J_rerr_ri_val = sf_J_rerr_ri.subs(sf_ri, sf.V3(pose_i["r"] + epsilon)) \
                                .subs(sf_rj, sf.V3(pose_j["r"] + epsilon)) \
                                .subs(sf_rij, sf.V3(rotmat_to_rotvec(pose_ij_meas["R"]) + epsilon)) \
                                .to_numpy()

    J_rerr_rj_val = sf_J_rerr_rj.subs(sf_ri, sf.V3(pose_i["r"] + epsilon)) \
                                .subs(sf_rj, sf.V3(pose_j["r"] + epsilon)) \
                                .subs(sf_rij, sf.V3(rotmat_to_rotvec(pose_ij_meas["R"]) + epsilon)) \
                                .to_numpy()

    return J_rerr_ri_val, J_rerr_rj_val
