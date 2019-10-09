/**
 * 
 */
package org.latlab.learner.geast.procedures;

import org.latlab.learner.geast.BicEvaluator;
import org.latlab.learner.geast.IModelWithScore;
import org.latlab.learner.geast.context.IProcedureContext;
import org.latlab.learner.geast.operators.GivenCandidate;
import org.latlab.learner.geast.operators.SearchCandidate;
import org.latlab.learner.geast.operators.SearchOperator;
import org.latlab.util.Evaluator;
import org.latlab.util.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * An iterative procedure for improve a candidate model.
 * 
 * <p>
 * It holds some {@link SearchOperator} and uses them to find the best model
 * that can be generated by each of them. In an iteration of this procedure, the
 * best among the models found by the operators is used for the next iteration.
 * This procedure stops when an iteration fails to improve the model by a
 * threshold. This procedure {@link #refine(SearchCandidate)} a candidate model
 * before it is used as the base model for the next iteration.
 * 
 * @author leonard
 * 
 */
public class IterativeProcedure implements Procedure {
	/**
	 * Context of the algorithm.
	 */
	protected final IProcedureContext context;

	/**
	 * Whether this procedure succeeded to find a better model.
	 */
	private boolean succeeded = true;

	/**
	 * Search operators to use for each iteration.
	 */
	private final List<? extends SearchOperator> operators;

	private static final Evaluator<SearchCandidate> EVALUATOR =
			new BicEvaluator();

	/**
	 * Constructs this procedure with a context and a single search operator.
	 * 
	 * @param context
	 *            context of the algorithm
	 * @param operator
	 *            the only operator used in each iteration
	 */
	public IterativeProcedure(IProcedureContext context, SearchOperator operator) {
		assert operator != null;

		this.context = context;
		this.operators = Collections.singletonList(operator);
	}

	/**
	 * Constructs this procedure with a context and a list of operators.
	 * 
	 * @param context
	 *            context of the algorithm
	 * @param operators
	 *            list of operators used in each iteration
	 */
	public IterativeProcedure(IProcedureContext context,
			List<? extends SearchOperator> operators) {
		assert operators.size() > 0;

		this.context = context;
		this.operators = operators;
	}

	public SearchCandidate run(final IModelWithScore base) {
		context.log().writeStartElementWithTime(name(), null);

		// StoppingCriterion criterion = getStoppingCriterion();
		// criterion.setCurrent(base);

		SearchCandidate current = new GivenCandidate(base);
		boolean stop = false;
		succeeded = false;

		do {
			Evaluator<SearchCandidate> evaluator =
					getEvaluator(current.estimation());
			Pair<SearchCandidate, Double> stepResult =
					step(current.estimation(), evaluator);
			SearchCandidate candidate = stepResult.first;

			if (candidate.isNew()
					&& candidate.estimation().BicScore()
							- current.estimation().BicScore() > context.threshold()) {

				context.log().writeElementWithCandidateToFile("step",
						candidate, true);

				// if it can find a better model, try to refine it and use it
				// for the next step
				candidate = refine(candidate, stepResult.second, evaluator);
				current = candidate;

				succeeded = true;
			} else {
				// stop if it can't find a better model
				stop = true;
			}
		} while (!stop);

		context.log().writeElementWithEstimationToFile("completed",
				current.estimation(), name(), false);
		context.log().writeEndElement(name());

		return current;
	}

	/**
	 * Proceed one iteration, and returns the best candidate found from the
	 * available search operators starting from the {@code base} estimation.
	 * 
	 * @param base
	 *            from which the operators search
	 * @return the best candidate found by the search operators
	 */
	protected Pair<SearchCandidate, Double> step(IModelWithScore base,
			Evaluator<SearchCandidate> evaluator) {
		// looks for a better candidate with each search operator
		List<SearchCandidate> candidates =
				new ArrayList<SearchCandidate>(operators.size());
		for (SearchOperator operator : operators) {
			SearchCandidate candidate = operator.search(base, evaluator);
			candidates.add(candidate);
			operator.update(candidate);
		}

		// finds the best candidate from the best of each operator. It is
		// separated from the search process so that the log can be clearer.
		double max = -Double.MAX_VALUE;
		SearchCandidate best = null;

		for (SearchCandidate candidate : candidates) {
			if (candidate.score() > max) {
				best = candidate;
				max = candidate.score();
			}

			context.log().writeElement("ranking", candidate, true);
		}

		if (best == null) {
			best = new GivenCandidate(base);
			max = Double.NaN;
		}

		return Pair.construct(best, max);
	}

	public boolean succeeded() {
		return succeeded;
	}

	/**
	 * Refines a model before it is used in the next iteration.
	 * 
	 * @param candidate
	 *            the best model found among the search operations in the last
	 *            iteration
	 * @param criterion
	 *            the criterion to stop on a candidate
	 * @return a refined model which is at least as best as the given candidate
	 *         model
	 */
	protected SearchCandidate refine(SearchCandidate candidate, double score,
			Evaluator<SearchCandidate> evaluator) {
		return candidate;
	}

	/**
	 * Returns a evaluator for evaluating the scores of different search
	 * operator. The default function returns a evaluating that returns the BIC
	 * scores of the best models found by different search operators.
	 * 
	 * @param current
	 *            the base model that the candidates in the search operator
	 *            generated from
	 * @return a comparator
	 */
	protected Evaluator<SearchCandidate> getEvaluator(IModelWithScore base) {
		return EVALUATOR;
	}

	public String name() {
		return getClass().getSimpleName();
	}
}
